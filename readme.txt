Network		param		epoch		dataset		loss
FC		1024, 512	300		flower		0.0148		conv_autoencoder_fc_flower_1024_512.pth
FC2								0.0089		conv_autoencoder_fc2_flower_1024_512.pth	


FC2: change sigmoid to relu, this is function, not a layer. 

	use relu seems better than sigmoid
	add lr_scheduler: not  good. change  patience?


Conv:
	orig: conv(3,16), conv(16,8)
	curr: conv(3,32), conv(32,16)	result similar, how to increase quality of recon?	


-----------------------------------------
train:
	python3 main.py --mode eval --arch FC --dataset flower --imwidth 64 --cnum 3

