<?php

$baseDir = __DIR__;
$inputDirs = [
	$baseDir.'/train',
	$baseDir.'/test',
];
$outputPath = $baseDir.'/corpus.txt';

$output = fopen($outputPath, 'wb');
if ($output === false)
	throw new RuntimeException('Unable to create corpus file: '.$outputPath);

$written = 0;

foreach ($inputDirs as $inputDir)
{
	if (!is_dir($inputDir))
		throw new RuntimeException('Input directory not found: '.$inputDir);

	$iterator = new RecursiveIteratorIterator(
		new RecursiveDirectoryIterator(
			$inputDir,
			FilesystemIterator::SKIP_DOTS
		)
	);

	foreach ($iterator as $file)
	{
		if (!$file->isFile() || strtolower($file->getExtension()) !== 'txt')
			continue;

		$text = file_get_contents($file->getPathname());
		if ($text === false)
			throw new RuntimeException('Unable to read file: '.$file->getPathname());

		$text = trim(preg_replace('/\s+/u', ' ', $text) ?? '');
		if ($text === '')
			continue;

		fwrite($output, $text.PHP_EOL);
		$written++;
	}
}

fclose($output);

echo "Corpus written to {$outputPath}\n";
echo "Samples: {$written}\n";
