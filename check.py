from pipeline2 import run_pipeline, PipelineConfig

cfg = PipelineConfig(target_sr=None, routing={"type": "threshold"})
report = run_pipeline(r"C:\Users\caspa\OneDrive\Desktop\Dsci\AR project\data\corpus_a\Ching-a-lings-jazz-bazaar_jukebox-36042_001_00-00-31.wav", "test_output3.wav", cfg)
print(report["decisions"])