<?php
// Attiva sviluppo locale
// php scripts/composer-local.php
// composer update
// 
// Torna allo stato pulito
// git checkout -- composer.json

use PHP2xAI\Models\Model;
use PHP2xAI\Tensor\Tensor;
use PHP2xAI\Runtime\PHP\Optimizers\Optimizer;

class SentimentModel extends Model
{
	public function __construct(?Optimizer $optimizer = null, int $hidden1 = 128, int $hidden2 = 64, $V = 30000, $L = 256, $embDimension = 512)
	{
		$this->embTable = Tensor::init([$V, $embDimension], 0.05);
		
		$this->W1 = Tensor::init([$embDimension, $hidden1], 0.05);
		$this->W2 = Tensor::init([$hidden1, $hidden2], 0.05);
		$this->W3 = Tensor::init([$hidden2, 2], 0.05);
		
		$this->b1 = Tensor::zeros([$hidden1]);
		$this->b2 = Tensor::zeros([$hidden2]);
		$this->b3 = Tensor::zeros([2]);
		
		parent::__construct($optimizer);
	}
	
	public function getEmbMeanPool(Tensor $x) : Tensor
	{
		$paddingMask = $x->paddingMask(0);

		$embeddings = $x->embeddings($this->embTable);
		$embeddings->setTrainable(false);
		$embeddings->setRequiresGrad(true);

		$encoded = $this->bertEncoder(
			$embeddings,
			8,
			$embeddings->shape[2] * 4,
			$paddingMask,
			0.1
		);
		$encoded->setTrainable(false);
		$encoded->setRequiresGrad(true);

		$embMeanPool = $encoded->meanPooling($paddingMask);
		$embMeanPool->setTrainable(false);
		$embMeanPool->setRequiresGrad(true);

		return $embMeanPool;
	}
	
	public function forward(Tensor $x) : Tensor
	{
		$embMeanPool = $this->getEmbMeanPool($x);
		
		$L1 = $embMeanPool->matmul($this->W1)->add($this->b1)->ReLU()->dropout(20);
		$L2 = $L1->matmul($this->W2)->add($this->b2)->ReLU()->dropout(20);
		$L3 = $L2->matmul($this->W3)->add($this->b3);
		
		return $L3;
	}
	
	public function output(Tensor $x) : Tensor
	{
		$embMeanPool = $this->getEmbMeanPool($x);
		
		$L1 = $embMeanPool->matmul($this->W1)->add($this->b1)->ReLU();
		$L2 = $L1->matmul($this->W2)->add($this->b2)->ReLU();
		$logits = $L2->matmul($this->W3)->add($this->b3);
		
		return $logits->softmax();
	}
	
	public function loss(Tensor $x, Tensor $y) : Tensor
	{
		$logits = $this->forward($x);
		
		return $logits->CELogitsLabelInt($y)->mean();
	}
}
