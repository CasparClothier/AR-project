from pipeline2 import build_router, build_stages, PipelineConfig

cfg = PipelineConfig()
router = build_router(cfg.routing)
stages = build_stages(cfg.stages)

print("Router:", router)
print("Stage names:", [s.name for s in stages])

decisions = router.decide(stats={}, mel_db=None)
print("Decisions:", decisions)

for stage in stages:
    would_run = decisions.get(stage.name, False)
    print(f"  {stage.name}: would_run={would_run}")