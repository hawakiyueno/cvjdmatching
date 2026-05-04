param(
    [string]$Annotations = ".\djinni_ner_annotations_openai_20k_cleaned_v6.jsonl",
    [string]$OutputDir = ".\artifacts\span_ner_roberta_base_openai_cleaned_v6",
    [Alias("Epoch")]
    [int]$Epochs = 8,
    [int]$TrainBatchSize = 2,
    [int]$EvalBatchSize = 2,
    [int]$GradAccum = 8,
    [int]$MaxDocuments = 0,
    [int]$NegativeSpanMultiplier = 5,
    [int]$MinNegativeSpans = 160,
    [string]$SplitManifest = ".\artifacts\fixed_split_cleaned_v4.json",
    [string]$CleanupInput = ".\djinni_ner_annotations_openai_20k_cleaned_v5.jsonl",
    [string]$CleanupOutput = ".\djinni_ner_annotations_openai_20k_cleaned_v6.jsonl",
    [switch]$RefreshCleanup,
    [string]$ResumeCheckpoint = "",
    [int]$ResumeAdditionalEpochs = 0,
    [string]$Device = "",
    [string]$PythonExe = ""
)

if ($PythonExe -eq "") {
    $VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $VenvPython) {
        $PythonExe = $VenvPython
    }
    else {
        $PythonExe = "python"
    }
}

if ($RefreshCleanup -or ((-not (Test-Path -LiteralPath $Annotations)) -and $Annotations -eq $CleanupOutput -and (Test-Path -LiteralPath $CleanupInput))) {
    & $PythonExe "clean_openai_annotations.py" "--input" $CleanupInput "--output" $CleanupOutput "--overwrite"
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$args = @(
    "train_span_ner.py",
    "--annotations", $Annotations,
    "--output-dir", $OutputDir,
    "--model-name", "roberta-base",
    "--epochs", "$Epochs",
    "--train-batch-size", "$TrainBatchSize",
    "--eval-batch-size", "$EvalBatchSize",
    "--gradient-accumulation-steps", "$GradAccum",
    "--learning-rate", "2e-5",
    "--weight-decay", "0.01",
    "--warmup-ratio", "0.1",
    "--max-length", "256",
    "--max-span-width", "8",
    "--max-chars-per-example", "1200",
    "--chunk-overlap-chars", "120",
    "--negative-span-multiplier", "$NegativeSpanMultiplier",
    "--min-negative-spans", "$MinNegativeSpans",
    "--classifier-hidden-dim", "256",
    "--width-embedding-dim", "32",
    "--split-manifest", $SplitManifest
)

if ($Device -ne "") {
    $args += @("--device", $Device)
}

if ($MaxDocuments -gt 0) {
    $args += @("--max-documents", "$MaxDocuments")
}

if ($ResumeCheckpoint -ne "") {
    $args += @("--resume-checkpoint", $ResumeCheckpoint)
}

if ($ResumeAdditionalEpochs -gt 0) {
    $args += @("--resume-additional-epochs", "$ResumeAdditionalEpochs")
}

& $PythonExe @args
