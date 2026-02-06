$ErrorActionPreference = "Stop"

$batchSizes = @(4, 8, 16)
$learningRates = @(5e-5)
$hiddenDims = @(256)
$epochs = 100

foreach ($bs in $batchSizes) {
  foreach ($lr in $learningRates) {
    foreach ($hd in $hiddenDims) {
      Write-Host "Running: batch_size=$bs, learning_rate=$lr, hidden_dim=$hd, epochs=$epochs"
      python train_gnn.py --epochs $epochs --batch-size $bs --learning-rate $lr --hidden-dim $hd
      if ($LASTEXITCODE -ne 0) {
        throw "Training failed for batch_size=$bs learning_rate=$lr hidden_dim=$hd"
      }
    }
  }
}
