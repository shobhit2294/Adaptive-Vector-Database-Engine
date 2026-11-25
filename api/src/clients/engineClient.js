import { Pool } from 'undici';

export class EngineClient {
  constructor(baseUrl, maxConnections = 32) {
    this.url = new URL(baseUrl);
    this.pool = new Pool(this.url.origin, {
      connections: maxConnections,
      keepAliveTimeout: 30000,
      keepAliveMaxTimeout: 60000,
    });
  }

  async request(method, path, bodyObj = null) {
    const headers = {};
    let body;
    if (bodyObj !== null) {
      headers['content-type'] = 'application/json';
      body = JSON.stringify(bodyObj);
    }
    const { statusCode, body: resBody } = await this.pool.request({
      path,
      method,
      headers,
      body,
    });
    const text = await resBody.text();
    let data = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch {
      data = { raw: text };
    }
    if (statusCode >= 400) {
      const err = new Error(data?.error || `Engine error HTTP ${statusCode}`);
      err.status = statusCode;
      err.payload = data;
      throw err;
    }
    return data;
  }

  health() { return this.request('GET', '/health'); }
  stats() { return this.request('GET', '/v1/stats'); }
  upsertVector(id, vector) { return this.request('POST', '/v1/vectors', { id: Number(id), vector }); }
  upsertBatch(items) { return this.request('POST', '/v1/vectors/batch', { items }); }
  getVector(id) { return this.request('GET', `/v1/vectors/${Number(id)}`); }
  deleteVector(id) { return this.request('DELETE', `/v1/vectors/${Number(id)}`); }
  search(payload) { return this.request('POST', '/v1/search', payload); }
  reconcileWahi() { return this.request('POST', '/v1/wahi/reconcile', {}); }
  setWahiPolicy(mode, ram_budget_bytes = 0) {
    return this.request('POST', '/v1/wahi/policy', { mode, ram_budget_bytes });
  }
  flush() { return this.request('POST', '/v1/admin/flush', {}); }
  compact() { return this.request('POST', '/v1/admin/compact', {}); }

  async close() {
    await this.pool.close();
  }
}
