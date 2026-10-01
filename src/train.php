<?php

// Activate local development
// php scripts/composer-local.php
// composer update
// 
//	Back to clean state
// git checkout -- composer.json

use PHP2xAI\Runtime\PHP\Datasets\TrainValidateDataset;
use PHP2xAI\Runtime\PHP\Datasets\HDF5Dataset;
use PHP2xAI\Runtime\PHP\Datasets\StreamFileDataset;
use PHP2xAI\Runtime\PHP\Optimizers\Adam;

ini_set('precision', 30);
ini_set('serialize_precision', 30);
ini_set("display_errors", "On");
ini_set("memory_limit", "20G");

include("../vendor/autoload.php");
include("model.php");

$path = "./DataLabelInt";
$outputPath = "./Output";

if (!@is_dir($outputPath))
	@mkdir($outputPath, 0777, true);

$dataset = new HDF5Dataset($path."/train.h5", 300);
$valDataset = new HDF5Dataset($path."/test.h5", 300);

// $dataset = new StreamFileDataset($path."/train.txt", 300);
// $valDataset = new StreamFileDataset($path."/test.txt", 300);

$tvDataset = new TrainValidateDataset($dataset, $valDataset);

$optimizer = new Adam(0.001, 0.9, 0.999);
$optimizer->setGradClip(1.0); // evita spike di gradiente che fanno risalire la loss
$model = new SentimentModel($optimizer, 256, 128, 30000, 128);

$epochsNumber = 40;

$model->setRuntime("CPP");
$model->setProvider("EIGEN");
$model->train($tvDataset, $epochsNumber, realpath(".")."/weights.json",1, realpath(".")."/profiler.log");