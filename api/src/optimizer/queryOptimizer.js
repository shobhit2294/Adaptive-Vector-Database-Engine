import { buildFilterWhereClause } from '../db/postgres.js';
import { optimizerPlanTotal, pgQueryDuration } from '../metrics/prometheus.js';

export class HybridQueryOptimizer {
  constructor(pgStore, engineClient, opts = {}) {
    this.pg = pgStore;
    this.engine = engineClient;
    this.prefilterMaxRows = opts.prefilterMaxRows ?? 2500;
    this.prefilterMaxSelectivity = opts.prefilterMaxSelectivity ?? 0.08;
    this.postfilterMaxExpand = opts.postfilterMaxExpand ?? 20;
  }

  async search({ vector, k = 10, ef_search = 0, filter = null, strategy = 'auto', bypass_cold_rerank = false }) {
    const { whereSql, params, hasFilter } = buildFilterWhereClause(filter || {});
    const tStart = performance.now();

    if (!hasFilter) {
      optimizerPlanTotal.inc({ plan: 'pure_vector' });
      const engRes = await this.engine.search({
        vector,
        k,
        ef_search,
        bypass_cold_rerank,
      });
      const ids = engRes.results.map((r) => r.id);
      const pg0 = performance.now();
      const metaMap = await this.pg.getByIds(ids);
      pgQueryDuration.observe({ operation: 'hydrate' }, (performance.now() - pg0) / 1000);

      const enriched = engRes.results.map((r) => ({
        ...r,
        record: metaMap.get(Number(r.id)) || null,
      }));
      return {
        results: enriched,
        query_plan: {
          plan: 'pure_vector',
          selectivity: 1.0,
          matching_rows: null,
          total_ms: performance.now() - tStart,
          engine_diagnostics: engRes.diagnostics,
        },
      };
    }

    // Step 1: Selectivity estimation in PostgreSQL
    const pgEst0 = performance.now();
    const [totalRows, matchingRows] = await Promise.all([
      this.pg.countAll(),
      this.pg.countFiltered(whereSql, params),
    ]);
    const pgEstMs = performance.now() - pgEst0;
    pgQueryDuration.observe({ operation: 'estimate_selectivity' }, pgEstMs / 1000);

    if (matchingRows === 0) {
      optimizerPlanTotal.inc({ plan: 'empty_filter' });
      return {
        results: [],
        query_plan: {
          plan: 'empty_filter',
          selectivity: 0.0,
          matching_rows: 0,
          total_rows: totalRows,
          pg_ms: pgEstMs,
          total_ms: performance.now() - tStart,
        },
      };
    }

    const selectivity = matchingRows / Math.max(totalRows, 1);
    const usePrefilter =
      strategy === 'prefilter' ||
      (strategy === 'auto' &&
        (matchingRows <= 500 ||
          (selectivity <= this.prefilterMaxSelectivity && matchingRows <= this.prefilterMaxRows)));

    if (usePrefilter) {
      optimizerPlanTotal.inc({ plan: 'prefilter_allowlist' });
      const pgId0 = performance.now();
      const allowedIds = await this.pg.fetchFilteredIds(whereSql, params, 10000);
      const pgIdMs = performance.now() - pgId0;
      pgQueryDuration.observe({ operation: 'fetch_allowlist' }, pgIdMs / 1000);

      const engRes = await this.engine.search({
        vector,
        k,
        ef_search,
        allowed_ids: allowedIds,
        bypass_cold_rerank,
      });

      const pgHyd0 = performance.now();
      const metaMap = await this.pg.getByIds(engRes.results.map((r) => r.id));
      pgQueryDuration.observe({ operation: 'hydrate' }, (performance.now() - pgHyd0) / 1000);

      return {
        results: engRes.results.map((r) => ({ ...r, record: metaMap.get(Number(r.id)) || null })),
        query_plan: {
          plan: 'prefilter_allowlist',
          selectivity,
          matching_rows: matchingRows,
          total_rows: totalRows,
          allowlist_size: allowedIds.length,
          pg_ms: pgEstMs + pgIdMs,
          total_ms: performance.now() - tStart,
          engine_diagnostics: engRes.diagnostics,
        },
      };
    }

    // Adaptive post-filtering with expansion factor proportional to 1 / selectivity
    const expandFactor = Math.min(
      this.postfilterMaxExpand,
      Math.max(2, Math.ceil(1.6 / Math.max(selectivity, 0.05)))
    );
    const expandedK = Math.min(Math.max(k * expandFactor, k * 2), 2000);

    const engRes = await this.engine.search({
      vector,
      k: expandedK,
      ef_search: Math.max(ef_search || 0, expandedK),
      bypass_cold_rerank,
    });

    const candIds = engRes.results.map((r) => r.id);
    const pgFilt0 = performance.now();
    const metaMap = await this.pg.filterCandidateIds(candIds, filter);
    const pgFiltMs = performance.now() - pgFilt0;
    pgQueryDuration.observe({ operation: 'postfilter' }, pgFiltMs / 1000);

    const surviving = [];
    for (const r of engRes.results) {
      const rec = metaMap.get(Number(r.id));
      if (rec) {
        surviving.push({ ...r, record: rec });
        if (surviving.length >= k) break;
      }
    }

    // Completeness fallback if post-filter yielded < k items while enough matching rows exist
    if (surviving.length < k && matchingRows > surviving.length && matchingRows <= 10000) {
      optimizerPlanTotal.inc({ plan: 'postfilter_fallback_prefilter' });
      const allowedIds = await this.pg.fetchFilteredIds(whereSql, params, 10000);
      const fallbackRes = await this.engine.search({
        vector,
        k,
        ef_search,
        allowed_ids: allowedIds,
        bypass_cold_rerank,
      });
      const fbMeta = await this.pg.getByIds(fallbackRes.results.map((r) => r.id));
      return {
        results: fallbackRes.results.map((r) => ({ ...r, record: fbMeta.get(Number(r.id)) || null })),
        query_plan: {
          plan: 'postfilter_fallback_prefilter',
          selectivity,
          matching_rows: matchingRows,
          total_rows: totalRows,
          expanded_k: expandedK,
          pg_ms: pgEstMs + pgFiltMs,
          total_ms: performance.now() - tStart,
          engine_diagnostics: fallbackRes.diagnostics,
        },
      };
    }

    optimizerPlanTotal.inc({ plan: 'adaptive_postfilter' });
    return {
      results: surviving,
      query_plan: {
        plan: 'adaptive_postfilter',
        selectivity,
        matching_rows: matchingRows,
        total_rows: totalRows,
        expanded_k: expandedK,
        pg_ms: pgEstMs + pgFiltMs,
        total_ms: performance.now() - tStart,
        engine_diagnostics: engRes.diagnostics,
      },
    };
  }
}
