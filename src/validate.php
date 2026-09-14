<?php

use PHP2xAI\Runtime\PHP\Datasets\StreamFileDataset;

ini_set('precision', 30);
ini_set('serialize_precision', 30);
ini_set('display_errors', 'On');
ini_set('memory_limit', '20G');

include('../vendor/autoload.php');
include('model.php');

$path = './DataLabelInt';
$valDataset = new StreamFileDataset($path . '/test.txt', 300);

$model = new SentimentModel();
$model->setRuntime('CPP');
$model->setProvider('EIGEN');
$model->loadModel('./model.json', './weights.json');

$correct = 0;
$total = 0;

$start = microtime(true);

foreach ($valDataset as $batch)
{
	foreach ($batch as [$x, $y])
	{
		$predicted = (int)$model->predictLabelInt($x);
		$target = (int)$y[0];

		if ($predicted === $target)
			$correct++;

		$total++;
	}
}

$elapsed = microtime(true) - $start;
$accuracy = $total > 0 ? ($correct / $total * 100) : 0;

echo "Test samples: $total\n";
echo "Correct: $correct\n";
echo 'Accuracy: ' . number_format($accuracy, 2) . " %\n";
echo 'Elapsed: ' . number_format($elapsed, 2) . " s\n";
