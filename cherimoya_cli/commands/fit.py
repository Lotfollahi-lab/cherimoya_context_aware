# cherimoya_cli fit command
# Author: Jacob Schreiber <jmschreiber91@gmail.com>


def add_parser(subparsers):
	parser = subparsers.add_parser("fit", help="Fit a Cherimoya model.")
	parser.add_argument("-p", "--parameters", type=str, required=True,
		help="A JSON file containing the parameters for fitting the model.")
	return parser


def run(args):
	import argparse
	import copy
	import os
	import sys
	import json

	os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'

	import torch
	torch.backends.cudnn.benchmark = True
	torch.set_float32_matmul_precision('high')

	from torch.optim import Muon
	from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

	from cherimoya import Cherimoya
	from cherimoya.io import PeakGenerator

	from tangermeme.io import extract_loci

	from . import evaluate as evaluate_cmd
	from ..defaults import default_fit_parameters
	from ..utils import merge_parameters

	parameters = merge_parameters(args.parameters, default_fit_parameters)
	if parameters['skip']:
		sys.exit()

	if parameters['verbose']:
		print("Training Chroms: ", parameters['training_chroms'])
		print("Vaidation Chroms: ", parameters['validation_chroms'])

		print("\nLoading peaks from: ", parameters['loci'])
		print("Loading negatives from: ", parameters['negatives'])
		print("Loading sequence from: ", parameters['sequences'])
		print("Loading signal from: ", parameters['signals'])
		print("Loading controls from: ", parameters['controls'])
		print("Loading exclusion list from: ", parameters['exclusion_lists'])
		print()


	###

	training_data = PeakGenerator(
		peaks=parameters['loci'],
		negatives=parameters['negatives'],
		sequences=parameters['sequences'],
		signals=parameters['signals'],
		controls=parameters['controls'],
		chroms=parameters['training_chroms'],
		in_window=parameters['in_window'],
		out_window=parameters['out_window'],
		max_jitter=parameters['max_jitter'],
		negative_ratio=parameters['negative_ratio'],
		reverse_complement=parameters['reverse_complement'],
		summits=parameters['summits'],
		exclusion_lists=parameters['exclusion_lists'],
		random_state=parameters['random_state'],
		batch_size=parameters['batch_size'],
		num_workers=parameters['num_workers'],
		verbose=parameters['verbose']
	)

	valid_data = extract_loci(
		sequences=parameters['sequences'],
		signals=parameters['signals'],
		in_signals=parameters['controls'],
		loci=parameters['loci'],
		chroms=parameters['validation_chroms'],
		in_window=parameters['in_window'],
		out_window=parameters['out_window'],
		max_jitter=0,
		exclusion_lists=parameters['exclusion_lists'],
		ignore=list('QWERYUIOPSDFHJKLZXVBNM'),
		verbose=parameters['verbose']
	)

	if parameters['verbose']:
		print("\nTraining Set Peaks: ", training_data.dataset.peak_sequences.shape[0])
		print("Training Set Negatives: ", training_data.dataset.negative_sequences.shape[0])
		print("Validation Set Size: ", valid_data[0].shape[0], "\n")


	if parameters['verbose']:
		print("Negative Ratio: 1:{:4.4} pos:neg\n".format(
			parameters['negative_ratio']))

	###

	if parameters['controls'] is not None:
		valid_sequences, valid_signals, valid_controls = valid_data
		n_control_tracks = len(parameters['controls'])
	else:
		valid_sequences, valid_signals = valid_data
		valid_controls = None
		n_control_tracks = 0

	trimming = (parameters['in_window'] - parameters['out_window']) // 2

	model = Cherimoya(
		n_filters=parameters['n_filters'],
		n_layers=parameters['n_layers'],
		n_outputs=len(parameters['signals']),
		n_control_tracks=n_control_tracks,
		expansion=parameters['expansion'],
		residual_scale=parameters['residual_scale'],
		single_count_output=parameters['single_count_output'],
		trimming=trimming,
		name=parameters['name'],
		verbose=parameters['verbose']
	).to(parameters['device'])

	if parameters['verbose']:
		print("Model has {} dilated layers and {} filters".format(
			parameters['n_layers'], parameters['n_filters'])
		)
		print("Model has {} trainable parameters.\n".format(
			sum(p.numel() for p in model.parameters() if p.requires_grad))
		)

	num_warmup_epochs = 5
	max_epochs = parameters['max_epochs']
	num_warmup_iters = len(training_data) * num_warmup_epochs
	num_decay_iters = len(training_data) * max(1, max_epochs - num_warmup_epochs)

	muon_params = []
	adam_params = []
	for name, p in model.named_parameters():
		if p.ndim == 2 and "weight" in name and name != "linear.weight":
			muon_params.append(p)
		else:
			adam_params.append(p)

	muon_optimizer = Muon(
		muon_params,
		lr=parameters['muon_lr'],
		weight_decay=parameters['muon_wd']
	)

	muon_warmup_scheduler = LinearLR(muon_optimizer, start_factor=0.01,
		total_iters=num_warmup_iters)
	muon_decay_scheduler = CosineAnnealingLR(muon_optimizer,
		T_max=num_decay_iters, eta_min=1e-5)
	muon_scheduler = SequentialLR(
		muon_optimizer,
		schedulers=[muon_warmup_scheduler, muon_decay_scheduler],
		milestones=[num_warmup_iters]
	)

	adam_optimizer = torch.optim.AdamW(
		adam_params,
		lr=parameters['adam_lr'],
		weight_decay=parameters['adam_wd']
	)
	adam_warmup_scheduler = LinearLR(adam_optimizer, start_factor=0.01,
		total_iters=num_warmup_iters)
	adam_decay_scheduler = CosineAnnealingLR(adam_optimizer,
		T_max=num_decay_iters, eta_min=1e-5)
	adam_scheduler = SequentialLR(
		adam_optimizer,
		schedulers=[adam_warmup_scheduler, adam_decay_scheduler],
		milestones=[num_warmup_iters]
	)

	if parameters['verbose']:
		print("Muon Optimizer: lr={}, wd={}".format(parameters['muon_lr'], 
													parameters['muon_wd']))
		print("AdamW Optimizer: lr={}, wd={}\n".format(parameters['adam_lr'],
													 parameters['adam_wd'])
		  
	###

	

	model.fit(training_data,
		muon_optimizer, adam_optimizer,
		muon_scheduler, adam_scheduler,
		X_valid=valid_sequences,
		X_ctl_valid=valid_controls,
		y_valid=valid_signals,
		max_epochs=max_epochs,
		batch_size=parameters['batch_size'],
		early_stopping=parameters['early_stopping'],
		dtype=parameters['dtype'],
		device=parameters['device'])

	### Evaluate Model

	model_name = parameters['name'] or model.name

	evaluate_parameters = copy.deepcopy(parameters)
	evaluate_parameters['chroms'] = parameters['validation_chroms']
	evaluate_parameters['max_jitter'] = 0
	evaluate_parameters['reverse_complement'] = False
	evaluate_parameters['model'] = model_name + '.torch'
	evaluate_parameters['performance_filename'] = model_name + '.performance.tsv'


	fname = "{}.evaluate.json".format(model_name)
	with open(fname, "w") as outfile:
		outfile.write(json.dumps(evaluate_parameters, sort_keys=True, indent=4))

	evaluate_cmd.run(argparse.Namespace(parameters=fname))
