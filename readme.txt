man.py:	
	train and evaluation script, several autoencoder class definition
en_de_cl.py:
	class Encoder:	a simple encoder class
	class Classifier: a simple classifier class
	class Decoder:	a simple decoder class
	class AutoencoderNet:
			an autoencoder class that contain the Encoder and Decoder
	class ClassfyNet:
			consists of an Encoder and Classifier for classification
			use the same Encoder as AutoencoderNet	
	
-----------------------6/5/24 add Encoder Decoder and Classifier seperately----
	the subnet parameter will be saved in seperate files for ClassifierNet
	hence redefining the network
dataset folder:
	mkdir -p /data/office31_new
train AE 2:
	python3 main.py --mode train --arch AE2 --dataset amazon --imwidth 64 --cnum 3 --dsroot /data/office31_new
	python3 main.py --mode eval --arch AE2 --dataset amazon --imwidth 64 --cnum 3 --paramfn autoencodernetconv_autoencoder.pth.pth --dsroot /data/office31_new

train classify net:
	python3 main.py --mode train_cl --arch CL --dataset amazon --imwidth 64 --cnum 3 --dsroot /media/student/isaacsim/office31/
	python3 main.py --mode eval_cl --arch CL --dataset amazon --imwidth 64 --cnum 3 --dsroot /data/office31_new --paramfn classifiernetclsnet.pth

status: cl training runs, cros-entropy, 31 classes, start loss 3.31
	81 epoch loss 0.0005
	testing...

-----------------------6/4/24 -----------------
train:
	python3 main.py --mode train --arch Conv --dataset amazon --imwidth 64 --cnum 3 --dsroot /data/office31
	python3 main.py --mode train --arch FC --dataset flower --imwidth 64 --cnum 3
	python3 main.py --mode train --arch Conv --dataset flower --imwidth 64 --cnum 3
	python3 main.py --mode eval --arch FC --dataset flower --imwidth 64 --cnum 3
eval:
	python3 main.py --mode eval --arch Conv --dataset amazon --imwidth 64 --cnum 3 -paramfn conv_autoencoder_conv_amazon_32_16.pth

-----------------------6/4/24 add office31 dataset code-----
dataset_office31.py
test_officedataset.py

--------------------------------------- exp results and notes ----------------------------------------------
Network		param		epoch		dataset		loss	output
FC		1024, 512	300		flower		0.0148		conv_autoencoder_fc_flower_1024_512.pth
FC2								0.0089		conv_autoencoder_fc2_flower_1024_512.pth	
Conv		(3,32) (32,16)  50		flower		0.0061	16,16,16
				121				0.0045
						fl128		0.0023 good 16, 32, 32
						amazon128	0.0014 good 16, 32, 32 conv_autoencoder_conv_amazon_32_16.pth
						image resolution prop to the resolution of the output of encoder
							
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

