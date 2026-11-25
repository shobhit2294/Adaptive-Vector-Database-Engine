import express from 'express';

export function createVectorRouter({ pg, engine, optimizer }) {
  const router = express.Router();

  router.post('/vectors', async (req, res, next) => {
    try {
      const { id, vector, external_key, category, price, tags, metadata } = req.body;
      if (id === undefined || !Array.isArray(vector) || vector.length === 0) {
        return res.status(400).json({ error: 'id and non-empty vector array are required' });
      }
      await pg.upsertRecord({ id, external_key, category, price, tags, metadata });
      try {
        await engine.upsertVector(id, vector);
      } catch (err) {
        // Compensating rollback in PG if engine rejects vector (e.g. dimension mismatch)
        await pg.deleteRecord(id).catch(() => {});
        throw err;
      }
      res.json({ status: 'upserted', id: Number(id) });
    } catch (err) {
      next(err);
    }
  });

  router.post('/vectors/batch', async (req, res, next) => {
    try {
      const { items } = req.body;
      if (!Array.isArray(items) || items.length === 0) {
        return res.status(400).json({ error: 'items array is required' });
      }
      await pg.upsertBatch(items);
      await engine.upsertBatch(items.map((x) => ({ id: Number(x.id), vector: x.vector })));
      res.json({ status: 'upserted_batch', count: items.length });
    } catch (err) {
      next(err);
    }
  });

  router.get('/vectors/:id', async (req, res, next) => {
    try {
      const id = Number(req.params.id);
      const [engData, metaMap] = await Promise.all([
        engine.getVector(id),
        pg.getByIds([id]),
      ]);
      res.json({
        id,
        vector: engData.vector,
        record: metaMap.get(id) || null,
      });
    } catch (err) {
      next(err);
    }
  });

  router.put('/vectors/:id', async (req, res, next) => {
    try {
      const id = Number(req.params.id);
      const { vector, external_key, category, price, tags, metadata } = req.body;
      if (!Array.isArray(vector) || vector.length === 0) {
        return res.status(400).json({ error: 'non-empty vector array is required' });
      }
      await pg.upsertRecord({ id, external_key, category, price, tags, metadata });
      await engine.upsertVector(id, vector);
      res.json({ status: 'updated', id });
    } catch (err) {
      next(err);
    }
  });

  router.delete('/vectors/:id', async (req, res, next) => {
    try {
      const id = Number(req.params.id);
      const [engRes, pgDeleted] = await Promise.all([
        engine.deleteVector(id),
        pg.deleteRecord(id),
      ]);
      res.json({
        id,
        deleted: Boolean(engRes.deleted || pgDeleted),
        engine_deleted: Boolean(engRes.deleted),
        postgres_deleted: pgDeleted,
      });
    } catch (err) {
      next(err);
    }
  });

  router.post('/search', async (req, res, next) => {
    try {
      const { vector, k = 10, ef_search = 0, filter = null, strategy = 'auto', bypass_cold_rerank = false } = req.body;
      if (!Array.isArray(vector) || vector.length === 0) {
        return res.status(400).json({ error: 'non-empty query vector is required' });
      }
      const out = await optimizer.search({ vector, k, ef_search, filter, strategy, bypass_cold_rerank });
      res.json(out);
    } catch (err) {
      next(err);
    }
  });

  router.get('/stats', async (_req, res, next) => {
    try {
      const [engineStats, pgRows] = await Promise.all([
        engine.stats(),
        pg.countAll(),
      ]);
      res.json({
        postgres_records: pgRows,
        engine: engineStats,
      });
    } catch (err) {
      next(err);
    }
  });

  router.post('/wahi/reconcile', async (_req, res, next) => {
    try {
      const out = await engine.reconcileWahi();
      res.json(out);
    } catch (err) {
      next(err);
    }
  });

  router.post('/wahi/policy', async (req, res, next) => {
    try {
      const { mode = 'wahi_adaptive', ram_budget_bytes = 0 } = req.body;
      const out = await engine.setWahiPolicy(mode, Number(ram_budget_bytes));
      res.json(out);
    } catch (err) {
      next(err);
    }
  });

  router.post('/admin/flush', async (_req, res, next) => {
    try {
      res.json(await engine.flush());
    } catch (err) {
      next(err);
    }
  });

  router.post('/admin/compact', async (_req, res, next) => {
    try {
      res.json(await engine.compact());
    } catch (err) {
      next(err);
    }
  });

  return router;
}
