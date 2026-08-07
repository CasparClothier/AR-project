from pipeline2 import run_pipeline, PipelineConfig

cfg = PipelineConfig(target_sr=None, routing={"type": "threshold"})
report = run_pipeline(r"C:\Users\caspa\OneDrive\Desktop\Dsci\AR project\data\corpus_a\A-rag-time-episode_jukebox-129854_001_00-00-30.wav", "test_output.wav", cfg)
print(report["decisions"])