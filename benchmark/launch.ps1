$env:PATH = "C:\Users\usuario\anaconda3;C:\Users\usuario\anaconda3\Library\bin;C:\Users\usuario\anaconda3\DLLs;" + $env:PATH
$env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"
$bench = $PSScriptRoot
$procs = @()
foreach ($P in "P6", "P5", "P4", "P3", "P2", "P1") {
    $procs += Start-Process -FilePath "C:\Users\usuario\anaconda3\python.exe" -ArgumentList "`"$bench\run_bench.py`" $P" `
        -WorkingDirectory $bench -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput "$bench\stdout_$P.txt" -RedirectStandardError "$bench\stderr_$P.txt"
}
"Lanzados: " + (($procs | ForEach-Object { "$($_.Id)" }) -join ", ")
$procs | Wait-Process
"Todos terminaron: " + (Get-Date)
