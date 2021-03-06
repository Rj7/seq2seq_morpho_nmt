import matplotlib
matplotlib.use('Agg')
import math
import random
import re
import time
import unicodedata
import logging

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable


FORMAT = '%(asctime)-15s %(message)s'
USE_CUDA = torch.cuda.is_available()

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 50

class Lang:
	def __init__(self, name):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: "SOS", 1: "EOS"}
		self.n_words = 2  # Count SOS and EOS

	def index_words(self, sentence):
		for word in sentence.split(' '):
			self.index_word(word)

	def index_word(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
	)


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
	s = unicode_to_ascii(s.lower().strip())
	s = re.sub(r"([.!?])", r" \1", s)
	s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
	return s


def read_langs(lang1, lang2, reverse=False):
	print("Reading lines...")

	# Read the file and split into lines
	lines = open('data/%s-%s.txt' % (lang1, lang2)).read().strip().split('\n')

	# Split every line into pairs and normalize
	pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

	# Reverse pairs, make Lang instances
	if reverse:
		pairs = [list(reversed(p)) for p in pairs]
		input_lang = Lang(lang2)
		output_lang = Lang(lang1)
	else:
		input_lang = Lang(lang1)
		output_lang = Lang(lang2)

	return input_lang, output_lang, pairs


def filter_pair(p):
	return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


def filter_pairs(pairs):
	return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(lang1_name, lang2_name, reverse=False):
	input_lang, output_lang, pairs = read_langs(lang1_name, lang2_name, reverse)
	logging.info("Read %s sentence pairs" % len(pairs))

	pairs = filter_pairs(pairs)
	logging.info("Trimmed to %s sentence pairs" % len(pairs))

	logging.info("Indexing words...")
	for pair in pairs:
		input_lang.index_words(pair[0])
		output_lang.index_words(pair[1])

	return input_lang, output_lang, pairs


# Return a list of indexes, one for each word in the sentence
def indexes_from_sentence(lang, sentence):
	return [lang.word2index[word] for word in sentence.split(' ')]


def variable_from_sentence(lang, sentence):
	indexes = indexes_from_sentence(lang, sentence)
	indexes.append(EOS_token)
	var = Variable(torch.LongTensor(indexes).view(-1, 1))
	#     print('var =', var)
	if USE_CUDA: var = var.cuda()
	return var


def variables_from_pair(pair):
	input_variable = variable_from_sentence(input_lang, pair[0])
	target_variable = variable_from_sentence(output_lang, pair[1])
	return (input_variable, target_variable)


def as_minutes(s):
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)


def time_since(since, percent):
	now = time.time()
	s = now - since
	es = s / (percent)
	rs = es - s
	return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def show_plot(points):
	plt.figure()
	fig, ax = plt.subplots()
	loc = ticker.MultipleLocator(base=0.2) # put ticks at regular intervals
	ax.yaxis.set_major_locator(loc)
	plt.plot(points)
	plt.savefig('losses')


def evaluate(sentence, max_length=MAX_LENGTH):
	input_variable = variable_from_sentence(input_lang, sentence)
	input_length = input_variable.size()[0]

	# Run through encoder
	encoder_hidden = encoder.init_hidden()
	encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

	# Create starting vectors for decoder
	decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
	decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
	if USE_CUDA:
		decoder_input = decoder_input.cuda()
		decoder_context = decoder_context.cuda()

	decoder_hidden = encoder_hidden

	decoded_words = []
	decoder_attentions = torch.zeros(max_length, max_length)

	# Run through decoder
	for di in range(max_length):
		decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context,
																					 decoder_hidden, encoder_outputs)
		decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

		# Choose top word from output
		topv, topi = decoder_output.data.topk(1)
		ni = topi[0][0]
		if ni == EOS_token:
			decoded_words.append('<EOS>')
			break
		else:
			decoded_words.append(output_lang.index2word[ni])

		# Next input is chosen word
		decoder_input = Variable(torch.LongTensor([[ni]]))
		if USE_CUDA: decoder_input = decoder_input.cuda()

	return decoded_words, decoder_attentions[:di + 1, :len(encoder_outputs)]


def evaluate_randomly():
	pair = random.choice(pairs)

	output_words, decoder_attn = evaluate(pair[0])
	output_sentence = ' '.join(output_words)

	logging.info('>{}'.format(pair[0]))
	logging.info('={}'.format(pair[1]))
	logging.info('<{}'.format(output_sentence))


def show_attention(input_sentence, output_words, attentions):
	# Set up figure with colorbar
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(attentions.numpy(), cmap='bone')
	fig.colorbar(cax)

	# Set up axes
	ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
	ax.set_yticklabels([''] + output_words)

	# Show label at every tick
	ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
	ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

	# plt.show()
	plt.savefig(input_sentence)
	plt.close()


def evaluate_and_show_attention(input_sentence):
	output_words, attentions = evaluate(input_sentence)
	logging.info('input = {}'.format(input_sentence))
	logging.info('output = {}'.format_map(' '.join(output_words)))
	show_attention(input_sentence, output_words, attentions)


class EncoderRNN(nn.Module):
	def __init__(self, input_size, hidden_size, n_layers=1):
		super(EncoderRNN, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers

		self.embedding = nn.Embedding(input_size, hidden_size)
		self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

	def forward(self, word_inputs, hidden):
		# Note: we run this all at once (over the whole input sequence)
		seq_len = len(word_inputs)
		embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
		output, hidden = self.gru(embedded, hidden)
		return output, hidden

	def init_hidden(self):
		hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
		if USE_CUDA: hidden = hidden.cuda()
		return hidden


class Attn(nn.Module):
	def __init__(self, method, hidden_size, max_length=MAX_LENGTH):
		super(Attn, self).__init__()

		self.method = method
		self.hidden_size = hidden_size

		if self.method == 'general':
			self.attn = nn.Linear(self.hidden_size, hidden_size)

		elif self.method == 'concat':
			self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
			self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

	def forward(self, hidden, encoder_outputs):
		seq_len = len(encoder_outputs)

		# Create variable to store attention energies
		attn_energies = Variable(torch.zeros(seq_len))  # B x 1 x S
		if USE_CUDA: attn_energies = attn_energies.cuda()

		# Calculate energies for each encoder output
		for i in range(seq_len):
			attn_energies[i] = self.score(hidden, encoder_outputs[i])

		# Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
		return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)

	def score(self, hidden, encoder_output):

		if self.method == 'dot':
			energy = hidden.dot(encoder_output)
			return energy

		elif self.method == 'general':
			energy = self.attn(encoder_output)
			energy = hidden.dot(energy)
			return energy

		elif self.method == 'concat':
			energy = self.attn(torch.cat((hidden, encoder_output), 1))
			energy = self.other.dot(energy)
			return energy


class AttnDecoderRNN(nn.Module):
	def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1):
		super(AttnDecoderRNN, self).__init__()

		# Keep parameters for reference
		self.attn_model = attn_model
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.dropout_p = dropout_p

		# Define layers
		self.embedding = nn.Embedding(output_size, hidden_size)
		self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
		self.out = nn.Linear(hidden_size * 2, output_size)

		# Choose attention model
		if attn_model != 'none':
			self.attn = Attn(attn_model, hidden_size)

	def forward(self, word_input, last_context, last_hidden, encoder_outputs):
		# Note: we run this one step at a time

		# Get the embedding of the current input word (last output word)
		word_embedded = self.embedding(word_input).view(1, 1, -1)  # S=1 x B x N

		# Combine embedded input word and last context, run through RNN
		rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
		rnn_output, hidden = self.gru(rnn_input, last_hidden)

		# Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
		attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
		context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N

		# Final output layer (next word prediction) using the RNN hidden state and context vector
		rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
		context = context.squeeze(1)  # B x S=1 x N -> B x N
		output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))

		# Return final output, hidden state, and attention weights (for visualization)
		return output, context, hidden, attn_weights


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
		  max_length=MAX_LENGTH):
	# Zero gradients of both optimizers
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()
	loss = 0  # Added onto for each word

	# Get size of input and target sentences
	input_length = input_variable.size()[0]
	target_length = target_variable.size()[0]

	# Run words through encoder
	encoder_hidden = encoder.init_hidden()
	encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

	# Prepare input and output variables
	decoder_input = Variable(torch.LongTensor([[SOS_token]]))
	decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
	decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder
	if USE_CUDA:
		decoder_input = decoder_input.cuda()
		decoder_context = decoder_context.cuda()

	# Choose whether to use teacher forcing
	use_teacher_forcing = random.random() < teacher_forcing_ratio
	if use_teacher_forcing:

		# Teacher forcing: Use the ground-truth target as the next input
		for di in range(target_length):
			decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context,
																						 decoder_hidden,
																						 encoder_outputs)
			loss += criterion(decoder_output, target_variable[di])
			decoder_input = target_variable[di]  # Next target is next input

	else:
		# Without teacher forcing: use network's own prediction as the next input
		for di in range(target_length):
			decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context,
																						 decoder_hidden,
																						 encoder_outputs)
			loss += criterion(decoder_output, target_variable[di])

			# Get most likely word index (highest value) from output
			topv, topi = decoder_output.data.topk(1)
			ni = topi[0][0]

			decoder_input = Variable(torch.LongTensor([[ni]]))  # Chosen word is next input
			if USE_CUDA: decoder_input = decoder_input.cuda()

			# Stop at end of sentence (not necessary when using known targets)
			if ni == EOS_token: break

	# Backpropagation
	loss.backward()
	torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
	torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.data[0] / target_length


# Configuring training
n_epochs = 50000
plot_every = 200
print_every = 1000
teacher_forcing_ratio = 0.5
clip = 5.0
attn_model = 'general'
hidden_size = 500
n_layers = 2
dropout_p = 0.05
learning_rate = 0.0001

# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0  # Reset every print_every
plot_loss_total = 0  # Reset every plot_every
source = 'de'
target = 'en'
logging.basicConfig(format=FORMAT, level=logging.INFO, filename="{}-{}.log".format(source, target))


if __name__ == "__main__":
	logging.info("Starting Program")
	input_lang, output_lang, pairs = prepare_data(source,target , True)
	logging.info(random.choice(pairs))

	# Initialize models
	encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers)
	decoder = AttnDecoderRNN(attn_model, hidden_size, output_lang.n_words, n_layers, dropout_p=dropout_p)

	# Move models to GPU
	if USE_CUDA:
		encoder.cuda()
		decoder.cuda()

	# Initialize optimizers and criterion
	encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
	decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
	criterion = nn.NLLLoss()

	for epoch in range(1, n_epochs + 1):

		# Get training data for this cycle
		training_pair = variables_from_pair(random.choice(pairs))
		input_variable = training_pair[0]
		target_variable = training_pair[1]

		# Run the train function
		loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

		# Keep track of loss
		print_loss_total += loss
		plot_loss_total += loss

		if epoch == 0: continue

		if epoch % print_every == 0:
			print_loss_avg = print_loss_total / print_every
			print_loss_total = 0
			print_summary = '%s (%d %d%%) %.4f' % (
				time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
			logging.info(print_summary)

		if epoch % plot_every == 0:
			plot_loss_avg = plot_loss_total / plot_every
			plot_losses.append(plot_loss_avg)
			plot_loss_total = 0

	show_plot(plot_losses)
	evaluate_randomly()

	evaluate_and_show_attention("elle a cinq ans de moins que moi .")
	evaluate_and_show_attention("elle est trop petit .")
	evaluate_and_show_attention("je ne crains pas de mourir .")
	evaluate_and_show_attention("c est un jeune directeur plein de talent .")
