import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import pg from 'pg';

const { Pool } = pg;
const __dirname = path.dirname(fileURLToPath(import.meta.url));

export function buildFilterWhereClause(filter = {}, startParamIdx = 1) {
  const clauses = [];
  const params = [];
  let idx = startParamIdx;

  if (filter.category !== undefined && filter.category !== null) {
    clauses.push(`category = $${idx++}`);
    params.push(String(filter.category));
  }
  if (Array.isArray(filter.categories) && filter.categories.length > 0) {
    clauses.push(`category = ANY($${idx++}::text[])`);
    params.push(filter.categories.map(String));
  }
  if (filter.min_price !== undefined && filter.min_price !== null) {
    clauses.push(`price >= $${idx++}`);
    params.push(Number(filter.min_price));
  }
  if (filter.max_price !== undefined && filter.max_price !== null) {
    clauses.push(`price <= $${idx++}`);
    params.push(Number(filter.max_price));
  }
  if (Array.isArray(filter.tags_all) && filter.tags_all.length > 0) {
    clauses.push(`tags @> $${idx++}::text[]`);
    params.push(filter.tags_all.map(String));
  }
  if (Array.isArray(filter.tags_any) && filter.tags_any.length > 0) {
    clauses.push(`tags && $${idx++}::text[]`);
    params.push(filter.tags_any.map(String));
  }
  if (filter.metadata && typeof filter.metadata === 'object' && Object.keys(filter.metadata).length > 0) {
    clauses.push(`metadata @> $${idx++}::jsonb`);
    params.push(JSON.stringify(filter.metadata));
  }

  const whereSql = clauses.length > 0 ? `WHERE ${clauses.join(' AND ')}` : '';
  return { whereSql, params, hasFilter: clauses.length > 0, nextParamIdx: idx };
}

export class PostgresStore {
  constructor(databaseUrl) {
    this.pool = new Pool({
      connectionString: databaseUrl,
      max: 20,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 5000,
    });
  }

  async initWithRetry(maxAttempts = 20, delayMs = 1500) {
    const schemaPath = path.join(__dirname, 'schema.sql');
    const schemaSql = await fs.readFile(schemaPath, 'utf8');
    let lastErr;
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        await this.pool.query(schemaSql);
        return;
      } catch (err) {
        lastErr = err;
        await new Promise((r) => setTimeout(r, delayMs));
      }
    }
    throw lastErr;
  }

  async health() {
    const res = await this.pool.query('SELECT 1 AS ok');
    return res.rows[0]?.ok === 1;
  }

  async upsertRecord(record) {
    const sql = `
      INSERT INTO vector_records (id, external_key, category, price, tags, metadata, updated_at)
      VALUES ($1, $2, $3, $4, $5, $6::jsonb, NOW())
      ON CONFLICT (id) DO UPDATE SET
        external_key = COALESCE(EXCLUDED.external_key, vector_records.external_key),
        category = EXCLUDED.category,
        price = EXCLUDED.price,
        tags = EXCLUDED.tags,
        metadata = EXCLUDED.metadata,
        updated_at = NOW()
    `;
    await this.pool.query(sql, [
      Number(record.id),
      record.external_key || null,
      record.category || 'general',
      Number(record.price ?? 0.0),
      Array.isArray(record.tags) ? record.tags.map(String) : [],
      JSON.stringify(record.metadata || {}),
    ]);
  }

  async upsertBatch(records) {
    if (!records.length) return;
    const ids = [];
    const keys = [];
    const cats = [];
    const prices = [];
    const tagsJson = [];
    const metas = [];
    for (const r of records) {
      ids.push(Number(r.id));
      keys.push(r.external_key || null);
      cats.push(r.category || 'general');
      prices.push(Number(r.price ?? 0.0));
      tagsJson.push(JSON.stringify(Array.isArray(r.tags) ? r.tags.map(String) : []));
      metas.push(JSON.stringify(r.metadata || {}));
    }
    const sql = `
      INSERT INTO vector_records (id, external_key, category, price, tags, metadata, updated_at)
      SELECT
        t.id,
        t.external_key,
        t.category,
        t.price,
        ARRAY(SELECT jsonb_array_elements_text(t.tags_json)),
        t.metadata,
        NOW()
      FROM UNNEST(
        $1::bigint[],
        $2::text[],
        $3::text[],
        $4::double precision[],
        $5::jsonb[],
        $6::jsonb[]
      ) AS t(id, external_key, category, price, tags_json, metadata)
      ON CONFLICT (id) DO UPDATE SET
        external_key = COALESCE(EXCLUDED.external_key, vector_records.external_key),
        category = EXCLUDED.category,
        price = EXCLUDED.price,
        tags = EXCLUDED.tags,
        metadata = EXCLUDED.metadata,
        updated_at = NOW()
    `;
    await this.pool.query(sql, [ids, keys, cats, prices, tagsJson, metas]);
  }

  async deleteRecord(id) {
    const res = await this.pool.query('DELETE FROM vector_records WHERE id = $1', [Number(id)]);
    return res.rowCount > 0;
  }

  async getByIds(ids) {
    if (!ids.length) return new Map();
    const numIds = ids.map((x) => Number(x));
    const res = await this.pool.query(
      'SELECT id, external_key, category, price, tags, metadata, created_at, updated_at FROM vector_records WHERE id = ANY($1::bigint[])',
      [numIds]
    );
    const map = new Map();
    for (const row of res.rows) {
      map.set(Number(row.id), {
        id: Number(row.id),
        external_key: row.external_key,
        category: row.category,
        price: row.price,
        tags: row.tags,
        metadata: row.metadata,
        created_at: row.created_at,
        updated_at: row.updated_at,
      });
    }
    return map;
  }

  async countAll() {
    const res = await this.pool.query('SELECT COUNT(*)::bigint AS cnt FROM vector_records');
    return Number(res.rows[0].cnt);
  }

  async countFiltered(whereSql, params) {
    const res = await this.pool.query(`SELECT COUNT(*)::bigint AS cnt FROM vector_records ${whereSql}`, params);
    return Number(res.rows[0].cnt);
  }

  async fetchFilteredIds(whereSql, params, limit = 10000) {
    const limIdx = params.length + 1;
    const res = await this.pool.query(
      `SELECT id FROM vector_records ${whereSql} ORDER BY id LIMIT $${limIdx}`,
      [...params, limit]
    );
    return res.rows.map((r) => Number(r.id));
  }

  async filterCandidateIds(candidateIds, filter) {
    if (!candidateIds.length) return new Map();
    const { whereSql, params, nextParamIdx } = buildFilterWhereClause(filter, 2);
    const idFilter = `id = ANY($1::bigint[])`;
    const combinedWhere = whereSql ? `${whereSql} AND ${idFilter}` : `WHERE ${idFilter}`;
    const res = await this.pool.query(
      `SELECT id, external_key, category, price, tags, metadata FROM vector_records ${combinedWhere}`,
      [candidateIds.map(Number), ...params]
    );
    const map = new Map();
    for (const row of res.rows) {
      map.set(Number(row.id), {
        id: Number(row.id),
        external_key: row.external_key,
        category: row.category,
        price: row.price,
        tags: row.tags,
        metadata: row.metadata,
      });
    }
    return map;
  }

  async close() {
    await this.pool.end();
  }
}
