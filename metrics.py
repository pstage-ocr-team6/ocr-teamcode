import editdistance
import numpy as np

def word_error_rate(predicted_outputs, ground_truths):
	sum_wer=0.0
	for output,ground_truth in zip(predicted_outputs,ground_truths):
		output=output.split(" ")
		ground_truth=ground_truth.split(" ")
		distance = editdistance.eval(output, ground_truth)
		length = max(len(output),len(ground_truth))
		sum_wer+=(distance/length)
	return sum_wer/len(predicted_outputs)


def sentence_acc(predicted_outputs, ground_truths):
	correct_sentences=0
	for output,ground_truth in zip(predicted_outputs,ground_truths):
		if np.array_equal(output,ground_truth):
			correct_sentences+=1
	return correct_sentences/len(predicted_outputs)


def get_worst_wer_img_path(img_path_list, predicted_outputs, ground_truths):
	max_wer_ind = 0
	max_wer = 0

	i = 0
	for output, ground_truth in zip(predicted_outputs,ground_truths):
		output=output.split(" ")
		ground_truth=ground_truth.split(" ")

		distance = editdistance.eval(output, ground_truth)
		length = max(len(output), len(ground_truth))
		cur_wer = (distance / length)
		if max_wer < cur_wer:
			max_wer = cur_wer
			max_wer_ind = i
		i+=1

	return img_path_list[max_wer_ind], max_wer, ground_truths[max_wer_ind], predicted_outputs[max_wer_ind]
