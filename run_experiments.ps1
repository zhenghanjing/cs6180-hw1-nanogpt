# CS6180 HW1 Q4 - all experiments.
# Seed (1337) and max_iters (5000) are fixed by config/train_shakespeare_char.py.
# Each run changes exactly ONE thing relative to the baseline.
#
# Usage (from the nanoGPT folder):
#   powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1

# Q4.1 baseline with the ORIGINAL nanoGPT code (run before modifying model.py):
#   python -u train.py config/train_shakespeare_char.py --compile=False --out_dir=out-baseline | Tee-Object logs/baseline.txt

$runs = [ordered]@{
  "baseline2" = @()                        # modified code, all switches at default = baseline architecture
  "rmsnorm"   = @("--norm_type=rmsnorm")   # Q4.2
  "swiglu"    = @("--mlp_type=swiglu")     # Q4.3
  "nope"      = @("--pos_type=none")       # Q4.4
  "rope"      = @("--pos_type=rope")       # Q4.4
  "gqa"       = @("--n_kv_head=3")         # Q4.5 (6 query heads, 3 kv heads -> group size 2)
}
New-Item -ItemType Directory -Force logs | Out-Null

# default learning rate (1e-3)
foreach ($name in $runs.Keys) {
  $extra = $runs[$name]
  Write-Host "===== $name =====" -ForegroundColor Cyan
  python -u train.py config/train_shakespeare_char.py --compile=False "--out_dir=out-$name" @extra | Tee-Object -FilePath "logs/$name.txt"
}

# learning-rate sweep (min_lr = lr / 10)
$lrs = [ordered]@{ "5e-4" = "5e-5"; "2e-3" = "2e-4" }
foreach ($lr in $lrs.Keys) {
  foreach ($name in $runs.Keys) {
    $extra = $runs[$name]
    $tag = "$name-lr$lr"
    Write-Host "===== $tag =====" -ForegroundColor Cyan
    python -u train.py config/train_shakespeare_char.py --compile=False "--out_dir=out-$tag" "--learning_rate=$lr" "--min_lr=$($lrs[$lr])" @extra | Tee-Object -FilePath "logs/$tag.txt"
  }
}

# plots
python plot.py fig_baseline.png logs/baseline.txt
python plot.py cmp_rmsnorm.png logs/baseline2.txt logs/rmsnorm.txt
python plot.py cmp_swiglu.png  logs/baseline2.txt logs/swiglu.txt
python plot.py cmp_pos.png     logs/baseline2.txt logs/nope.txt logs/rope.txt
python plot.py cmp_gqa.png     logs/baseline2.txt logs/gqa.txt
