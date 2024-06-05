Network		param		epoch		dataset		loss	output
FC		1024, 512	300		flower		0.0148		conv_autoencoder_fc_flower_1024_512.pth
FC2								0.0089		conv_autoencoder_fc2_flower_1024_512.pth	
Conv		(3,32) (32,16)  50		flower		0.0061	16,16,16
				121				0.0045
						fl128		0.0023 good 16, 32, 32
		(3,32)(32,16) no maxpool	flower		0.0012	16,32,32 near perfect
			
					after 50 epochs, little improv
		(3,32) (32,24)					0.0036   24,16,16

FC2: change sigmoid to relu, this is function, not a layer. 

	use relu seems better than sigmoid
	add lr_scheduler: not  good. change  patience?


Conv:
	orig: conv(3,16), conv(16,8)
	curr: conv(3,32), conv(32,16)	result similar, how to increase quality of recon?	


-----------------------------------------
train:
	python3 main.py --mode train --arch FC --dataset flower --imwidth 64 --cnum 3
	python3 main.py --mode train --arch Conv --dataset flower --imwidth 64 --cnum 3
	python3 main.py --mode eval --arch FC --dataset flower --imwidth 64 --cnum 3

-----------------------6/4/24 add office31 dataset code-----
dataset_office31.py
test_officedataset.py

