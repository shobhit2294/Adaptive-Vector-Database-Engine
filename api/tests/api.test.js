import test from 'node:test';
import assert from 'node:assert/strict';
import { buildFilterWhereClause } from '../src/db/postgres.js';
import { HybridQueryOptimizer } from '../src/optimizer/queryOptimizer.js';

test('buildFilterWhereClause builds parameterized SQL safely', () => {
  const { whereSql, params, hasFilter } = buildFilterWhereClause({
    category: 'electronics',
    min_price: 50,
    max_price: 250,
    tags_all: ['wireless', 'audio'],
    metadata: { brand: 'acme' },
  });
  assert.equal(hasFilter, true);
  assert.match(whereSql, /category = \$1/);
  assert.match(whereSql, /price >= \$2/);
  assert.match(whereSql, /price <= \$3/);
  assert.match(whereSql, /tags @> \$4::text\[\]/);
  assert.match(whereSql, /metadata @> \$5::jsonb/);
  assert.equal(params.length, 5);
});

test('HybridQueryOptimizer selects prefilter when selectivity is tight', async () => {
  let receivedSearchPayload = null;
  const mockPg = {
    async countAll() { return 10000; },
    async countFiltered() { return 120; }, // 1.2% selectivity -> prefilter
    async fetchFilteredIds() { return [11, 22, 33]; },
    async getByIds(ids) {
      const m = new Map();
      for (const id of ids) m.set(id, { id, category: 'books', price: 15 });
      return m;
    },
  };
  const mockEngine = {
    async search(payload) {
      receivedSearchPayload = payload;
      return {
        results: [{ id: 22, distance: 0.1, score: 0.9, tier: 'hot_ram' }],
        diagnostics: { latency_ms: 0.5 },
      };
    },
  };

  const opt = new HybridQueryOptimizer(mockPg, mockEngine);
  const out = await opt.search({
    vector: [0.1, 0.2, 0.3, 0.4],
    k: 5,
    filter: { category: 'books' },
  });

  assert.equal(out.query_plan.plan, 'prefilter_allowlist');
  assert.deepEqual(receivedSearchPayload.allowed_ids, [11, 22, 33]);
  assert.equal(out.results.length, 1);
  assert.equal(out.results[0].record.category, 'books');
});

test('HybridQueryOptimizer selects adaptive postfilter when selectivity is broad', async () => {
  let receivedSearchPayload = null;
  const mockPg = {
    async countAll() { return 10000; },
    async countFiltered() { return 4000; }, // 40% selectivity -> adaptive postfilter
    async filterCandidateIds(ids) {
      const m = new Map();
      for (const id of ids) {
        if (id % 2 === 0) m.set(id, { id, category: 'home', price: 30 });
      }
      return m;
    },
  };
  const mockEngine = {
    async search(payload) {
      receivedSearchPayload = payload;
      return {
        results: [
          { id: 1, distance: 0.05, score: 0.95, tier: 'hot_ram' },
          { id: 2, distance: 0.08, score: 0.92, tier: 'hot_ram' },
          { id: 4, distance: 0.12, score: 0.88, tier: 'hot_ram' },
        ],
        diagnostics: { latency_ms: 0.7 },
      };
    },
  };

  const opt = new HybridQueryOptimizer(mockPg, mockEngine);
  const out = await opt.search({
    vector: [0.1, 0.2, 0.3, 0.4],
    k: 2,
    filter: { category: 'home' },
  });

  assert.equal(out.query_plan.plan, 'adaptive_postfilter');
  assert.equal(receivedSearchPayload.allowed_ids, undefined);
  assert.ok(receivedSearchPayload.k >= 4);
  assert.deepEqual(out.results.map((r) => r.id), [2, 4]);
});
