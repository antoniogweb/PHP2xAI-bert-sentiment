<?php

use PHP2xAI\Tokenizer\PHP\Tokenizer;

include("../vendor/autoload.php");

const SEQUENCE_LENGTH = 1024;
const TRAIN_SHUFFLE_SEED = 42;
const TEST_SHUFFLE_SEED = 43;

$baseDir = __DIR__;
$tokenizerPath = $baseDir.'/tokenizer.json';
$outputDir = $baseDir.'/DataLabelInt';

if (!is_file($tokenizerPath))
	throw new RuntimeException('Tokenizer file not found: '.$tokenizerPath);

if (!is_dir($outputDir) && !mkdir($outputDir, 0777, true) && !is_dir($outputDir))
	throw new RuntimeException('Unable to create output directory: '.$outputDir);

$tokenizer = new Tokenizer($tokenizerPath);

$trainSamples = writeDataset(
	$tokenizer,
	$baseDir.'/train',
	$outputDir.'/train.txt',
	TRAIN_SHUFFLE_SEED
);

$testSamples = writeDataset(
	$tokenizer,
	$baseDir.'/test',
	$outputDir.'/test.txt',
	TEST_SHUFFLE_SEED
);

echo "Train dataset written to {$outputDir}/train.txt\n";
echo "Train samples: {$trainSamples}\n";
echo "Test dataset written to {$outputDir}/test.txt\n";
echo "Test samples: {$testSamples}\n";

function writeDataset(Tokenizer $tokenizer, string $splitDir, string $outputPath, int $shuffleSeed): int
{
	if (!is_dir($splitDir))
		throw new RuntimeException('Input directory not found: '.$splitDir);

	$rows = [];

	foreach ([
		'negative' => 0,
		'positive' => 1,
	] as $labelDir => $label)
	{
		$dir = $splitDir.'/'.$labelDir;

		if (!is_dir($dir))
			throw new RuntimeException('Label directory not found: '.$dir);

		foreach (listTextFiles($dir) as $path)
		{
			$text = file_get_contents($path);
			if ($text === false)
				throw new RuntimeException('Unable to read file: '.$path);

			$text = trim(preg_replace('/\s+/u', ' ', $text) ?? '');
			if ($text === '')
				continue;

			$ids = $tokenizer->encodeFixed($text, SEQUENCE_LENGTH);
			$rows[] = implode(' ', $ids).'|'.$label;
		}
	}

	shuffleDeterministic($rows, $shuffleSeed);

	$output = fopen($outputPath, 'wb');
	if ($output === false)
		throw new RuntimeException('Unable to create dataset file: '.$outputPath);

	foreach ($rows as $row)
		fwrite($output, $row.PHP_EOL);

	fclose($output);

	return count($rows);
}

/**
 * @return string[]
 */
function listTextFiles(string $dir): array
{
	$files = [];

	$iterator = new RecursiveIteratorIterator(
		new RecursiveDirectoryIterator(
			$dir,
			FilesystemIterator::SKIP_DOTS
		)
	);

	foreach ($iterator as $file)
	{
		if ($file->isFile() && strtolower($file->getExtension()) === 'txt')
			$files[] = $file->getPathname();
	}

	sort($files, SORT_NATURAL);

	return $files;
}

/**
 * @param array<int,mixed> $items
 */
function shuffleDeterministic(array &$items, int $seed): void
{
	mt_srand($seed);

	for ($i = count($items) - 1; $i > 0; $i--)
	{
		$j = mt_rand(0, $i);

		if ($i === $j)
			continue;

		$tmp = $items[$i];
		$items[$i] = $items[$j];
		$items[$j] = $tmp;
	}
}
