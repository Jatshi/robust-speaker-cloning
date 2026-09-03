$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSScriptRoot
$Fixture = Join-Path $ProjectDir "outputs/smoke-fixture"
$Output = Join-Path $ProjectDir "outputs/smoke-checkpoints"
$env:PYTHONPATH = $ProjectDir
Set-Location $ProjectDir
python -m pytest -q
python -m src.make_smoke_fixture --output $Fixture
python -m src.train_v2 --manifest "$Fixture/manifest.json" --teacher-cache "$Fixture/teacher.pt" --feature-cache "$Fixture/feature_cache" --output $Output --epochs 1 --batch-size 2 --workers 0 --max-train-batches 1 --max-validation-batches 1 --memory-bank-size 8 --device cpu --allow-cpu-smoke
python -c "import torch; p=torch.load(r'$Output/best.pt', map_location='cpu', weights_only=True); assert p['epoch']==1 and p['queue']['values'].shape[0]==2; print('CPU pipeline smoke passed')"
