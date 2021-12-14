import numpy as np
import tensorflow as tf
import copy
from tensorflow.keras.models import Model
#tf.config.experimental_run_functions_eagerly(True)

'''
class ConvLSTMCell(tf.keras.Model):
    def __init__(self, initcell_input, inithidden_input, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.initcell = initcell_input
        self.inithidden = inithidden_input
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias

        self.conv = tf.keras.layers.Conv2D(
            filters=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding='same',
            use_bias=self.bias
        )

    @tf.function
    def call(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = tf.concat([input_tensor, h_cur], axis=3)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = tf.split(combined_conv, num_or_size_splits=4, axis=-1)
        i = tf.keras.activations.sigmoid(cc_i)
        f = tf.keras.activations.sigmoid(cc_f)
        o = tf.keras.activations.sigmoid(cc_o)
        g = tf.keras.activations.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * tf.keras.activations.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        print("262 line:",batch_size)
        return (tf.zeros([batch_size, height, width, self.hidden_dim]), tf.zeros([batch_size, height, width, self.hidden_dim]))
        #return (self.inithidden,self.initcell)

class Encoder(tf.keras.Model):
    def __init__(self,initcell_input,inithidden_input, hidden, enc_num_layers=1):
        super(Encoder, self).__init__()
        self.initcell = initcell_input
        self.inithidden = inithidden_input
        self.enc_num_layers = enc_num_layers
        self.encoder_input_convlstm = ConvLSTMCell(
            initcell_input = self.initcell,
            inithidden_input = self.inithidden,
            hidden_dim=hidden,
            kernel_size=(3, 3),
            bias=True
        )
        if self.enc_num_layers is not None:
            self.hidden_encoder_layers = [
                ConvLSTMCell(
                    initcell_input=self.initcell,
                    inithidden_input=self.inithidden,
                    hidden_dim=hidden,
                    kernel_size=(3, 3),
                    bias=True
                ) for _ in range(self.enc_num_layers)
            ]
    @tf.function
    def call(self, enc_input):
        h_t, c_t = self.init_hidden(enc_input, 'seq')
        if self.enc_num_layers is not None:
            hidden_h_t = []
            hidden_c_t = []
            for i in range(self.enc_num_layers):
                hidden_h_t += [self.init_hidden(h_t, i)[0]]
                hidden_c_t += [self.init_hidden(h_t, i)[1]]

        seq_len = enc_input.shape[1]
        for t in range(seq_len):
            h_t, c_t = self.encoder_input_convlstm.call(
                input_tensor=enc_input[:, t, :, :, :],
                cur_state=[h_t, c_t]
            )
            input_tensor = h_t
            if self.enc_num_layers is not None:
                for i in range(self.enc_num_layers):
                    hidden_h_t[i], hidden_c_t[i] = self.hidden_encoder_layers[i].call(
                        input_tensor=input_tensor,
                        cur_state=[hidden_h_t[i], hidden_c_t[i]]
                    )
                    input_tensor = hidden_h_t[i]

        if self.enc_num_layers is not None:
            return hidden_h_t[-1], hidden_c_t[-1]
        else:
            return h_t, c_t

    def init_hidden(self, input_tensor, seq):
        if seq == 'seq':
            #print("line317:",input_tensor.shape)
            #b, seq_len, h, w, _ = input_tensor.shape
            #h_t, c_t = self.encoder_input_convlstm.init_hidden(
            #    batch_size=b,
            #    image_size=(h, w)
            #)
            h_t = self.encoder_input_convlstm.inithidden
            c_t = self.encoder_input_convlstm.initcell
        else:
            #b, h, w, _ = input_tensor.shape
            #h_t, c_t = self.hidden_encoder_layers[seq].init_hidden(
            #    batch_size=b,
            #    image_size=(h, w)
            #)
            h_t = self.hidden_encoder_layers[seq].inithidden
            c_t = self.hidden_encoder_layers[seq].initcell

        return h_t, c_t

class Decoder(tf.keras.Model):
    def __init__(self,initcell_input,inithidden_input, hidden, dec_num_layers=1, future_len=12):
        super(Decoder, self).__init__()
        self.initcell = initcell_input
        self.inithidden = inithidden_input
        self.dec_num_layers = dec_num_layers
        self.future_len = future_len
        self.decoder_input_convlstm = ConvLSTMCell(
            initcell_input=self.initcell,
            inithidden_input=self.inithidden,
            hidden_dim=hidden,
            kernel_size=(3, 3),
            bias=True
        )
        if self.dec_num_layers is not None:
            self.hidden_decoder_layers = [
                ConvLSTMCell(
                    initcell_input=self.initcell,
                    inithidden_input=self.inithidden,
                    hidden_dim=hidden,
                    kernel_size=(3, 3),
                    bias=True
                ) for _ in range(dec_num_layers)
            ]

        self.decoder_output_layer = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(3, 3),
            padding='same',
            activation='sigmoid'
        )

    @tf.function
    def call(self, enc_output):
        if self.dec_num_layers is not None:
            hidden_h_t = []
            hidden_c_t = []
            for i in range(self.dec_num_layers):
                hidden_h_t += [self.init_hidden(enc_output[0], i)[0]]
                hidden_c_t += [self.init_hidden(enc_output[0], i)[1]]

        outputs = []
        input_tensor = enc_output[0]
        h_t, c_t = self.init_hidden(input_tensor, 'seq')
        for t in range(self.future_len):
            h_t, c_t = self.decoder_input_convlstm.call(
                input_tensor=input_tensor,
                cur_state=[h_t, c_t]
            )
            input_tensor = h_t
            if self.dec_num_layers is not None:
                for i in range(self.dec_num_layers):
                    hidden_h_t[i], hidden_c_t[i] = self.hidden_decoder_layers[i].call(
                        input_tensor=input_tensor,
                        cur_state=[hidden_h_t[i], hidden_c_t[i]]
                    )
                    input_tensor = hidden_h_t[i]
                output = self.decoder_output_layer(hidden_h_t[-1])
            else:
                output = self.decoder_output_layer(h_t)
            outputs += [output]
        outputs = tf.stack(outputs, 1)

        return outputs

    def init_hidden(self, input_tensor, seq):
        if seq == 'seq':
            #b, h, w, _ = input_tensor.shape
            #h_t, c_t = self.decoder_input_convlstm.init_hidden(
            #    batch_size=b,
            #    image_size=(h, w)
            #)
            h_t = self.decoder_input_convlstm.inithidden
            c_t = self.decoder_input_convlstm.initcell
        else:
            #b, h, w, _ = input_tensor.shape
            #h_t, c_t = self.hidden_decoder_layers[seq].init_hidden(
            #    batch_size=b,
            #    image_size=(h, w)
            #)
            h_t = self.hidden_decoder_layers[seq].inithidden
            c_t = self.hidden_decoder_layers[seq].initcell
        return h_t, c_t

class Seq2Seq(tf.keras.Model):
    def __init__(self, ecell,ehidden,dcell,dhidden,hidden, enc_num_layers=1, dec_num_layers=1,future_len = 10):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(initcell_input=ecell,inithidden_input=ehidden, hidden=hidden, enc_num_layers=enc_num_layers)
        self.decoder = Decoder(initcell_input=dcell,inithidden_input=dhidden, hidden=hidden, dec_num_layers=dec_num_layers,future_len=future_len)

    @tf.function
    def call(self, ecell,ehidden,dcell,dhidden,enc_input):
        enc_output = self.encoder(enc_input)
        dec_output = self.decoder(enc_output)
        return dec_output
'''

class ConvLSTMCell(tf.keras.Model):
    def __init__(self, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.bias = bias

        self.conv = tf.keras.layers.Conv2D(
            filters=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding='same',
            use_bias=self.bias
        )

    def call(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = tf.concat([input_tensor, h_cur], axis=3)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = tf.split(combined_conv, num_or_size_splits=4, axis=-1)
        i = tf.keras.activations.sigmoid(cc_i)
        f = tf.keras.activations.sigmoid(cc_f)
        o = tf.keras.activations.sigmoid(cc_o)
        g = tf.keras.activations.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * tf.keras.activations.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (tf.zeros([batch_size, height, width, self.hidden_dim]),
                tf.zeros([batch_size, height, width, self.hidden_dim]))

class Encoder(tf.keras.Model):
    def __init__(self, hidden, enc_num_layers=1):
        super(Encoder, self).__init__()
        self.enc_num_layers = enc_num_layers
        self.encoder_input_convlstm = ConvLSTMCell(
            hidden_dim=hidden,
            kernel_size=(3, 3),
            bias=True
        )
        if self.enc_num_layers is not None:
            self.hidden_encoder_layers = [
                ConvLSTMCell(
                    hidden_dim=hidden,
                    kernel_size=(3, 3),
                    bias=True
                ) for _ in range(self.enc_num_layers)
            ]

    def call(self, enc_input):
        h_t, c_t = self.init_hidden(enc_input, 'seq')
        if self.enc_num_layers is not None:
            hidden_h_t = []
            hidden_c_t = []
            for i in range(self.enc_num_layers):
                hidden_h_t += [self.init_hidden(h_t, i)[0]]
                hidden_c_t += [self.init_hidden(h_t, i)[1]]

        seq_len = enc_input.shape[1]
        for t in range(seq_len):
            h_t, c_t = self.encoder_input_convlstm(
                input_tensor=enc_input[:, t, :, :, :],
                cur_state=[h_t, c_t]
            )
            input_tensor = h_t
            if self.enc_num_layers is not None:
                for i in range(self.enc_num_layers):
                    hidden_h_t[i], hidden_c_t[i] = self.hidden_encoder_layers[i](
                        input_tensor=input_tensor,
                        cur_state=[hidden_h_t[i], hidden_c_t[i]]
                    )
                    input_tensor = hidden_h_t[i]

        if self.enc_num_layers is not None:
            return hidden_h_t[-1], hidden_c_t[-1]
        else:
            return h_t, c_t

    def init_hidden(self, input_tensor, seq):
        if seq == 'seq':
            b, seq_len, h, w, _ = input_tensor.shape
            h_t, c_t = self.encoder_input_convlstm.init_hidden(
                batch_size=b,
                image_size=(h, w)
            )
        else:
            b, h, w, _ = input_tensor.shape
            h_t, c_t = self.hidden_encoder_layers[seq].init_hidden(
                batch_size=b,
                image_size=(h, w)
            )
        return h_t, c_t

class Decoder(tf.keras.Model):
    def __init__(self, hidden, dec_num_layers=1, future_len=12):
        super(Decoder, self).__init__()
        self.dec_num_layers = dec_num_layers
        self.future_len = future_len
        self.decoder_input_convlstm = ConvLSTMCell(
            hidden_dim=hidden,
            kernel_size=(3, 3),
            bias=True
        )
        if self.dec_num_layers is not None:
            self.hidden_decoder_layers = [
                ConvLSTMCell(
                    hidden_dim=hidden,
                    kernel_size=(3, 3),
                    bias=True
                ) for _ in range(dec_num_layers)
            ]

        self.decoder_output_layer = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(3, 3),
            padding='same',
            activation='sigmoid'
        )

    def call(self, enc_output):
        if self.dec_num_layers is not None:
            hidden_h_t = []
            hidden_c_t = []
            for i in range(self.dec_num_layers):
                hidden_h_t += [self.init_hidden(enc_output[0], i)[0]]
                hidden_c_t += [self.init_hidden(enc_output[0], i)[1]]

        outputs = []
        input_tensor = enc_output[0]
        h_t, c_t = self.init_hidden(input_tensor, 'seq')
        for t in range(self.future_len):
            h_t, c_t = self.decoder_input_convlstm(
                input_tensor=input_tensor,
                cur_state=[h_t, c_t]
            )
            input_tensor = h_t
            if self.dec_num_layers is not None:
                for i in range(self.dec_num_layers):
                    hidden_h_t[i], hidden_c_t[i] = self.hidden_decoder_layers[i](
                        input_tensor=input_tensor,
                        cur_state=[hidden_h_t[i], hidden_c_t[i]]
                    )
                    input_tensor = hidden_h_t[i]
                output = self.decoder_output_layer(hidden_h_t[-1])
            else:
                output = self.decoder_output_layer(h_t)
            outputs += [output]
        outputs = tf.stack(outputs, 1)

        return outputs

    def init_hidden(self, input_tensor, seq):
        if seq == 'seq':
            b, h, w, _ = input_tensor.shape
            h_t, c_t = self.decoder_input_convlstm.init_hidden(
                batch_size=b,
                image_size=(h, w)
            )
        else:
            b, h, w, _ = input_tensor.shape
            h_t, c_t = self.hidden_decoder_layers[seq].init_hidden(
                batch_size=b,
                image_size=(h, w)
            )
        return h_t, c_t

class Seq2Seq(tf.keras.Model):
    def __init__(self, hidden, enc_num_layers=1, dec_num_layers=1, future_len=10):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(hidden, enc_num_layers)
        self.decoder = Decoder(hidden, dec_num_layers,future_len=future_len)

    def call(self, enc_input):
        enc_output = self.encoder(enc_input)
        dec_output = self.decoder(enc_output)

        return dec_output

class DNNpredict(tf.keras.Model):
    def __init__(self, layerDim=[42, 4000, 2000, 1]):
        super(DNNpredict, self).__init__()
        #self.d1 = tf.keras.layers.Dense(layerDim[1],activation='relu')#dropout,
        #self.d2 = tf.keras.layers.Dense(layerDim[2],activation='relu')#
        self.d1 = tf.keras.layers.Dense(2000,activation='relu')#dropout,
        self.d2 = tf.keras.layers.Dense(1000,activation='relu')#

        self.d3 = tf.keras.layers.Dense(layerDim[3],activation='relu')

    @tf.function
    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)

class DNNpredictLite(tf.keras.Model):
    def __init__(self, layerDim=[42, 4000, 2000, 1]):
        super(DNNpredictLite, self).__init__()
        #self.d1 = tf.keras.layers.Dense(layerDim[1],activation='relu')#dropout,
        #self.d2 = tf.keras.layers.Dense(layerDim[2],activation='relu')#
        #self.d1 = tf.keras.layers.Dense(2000,activation='relu')#dropout,
        self.d2 = tf.keras.layers.Dense(1000,activation='relu')#

        self.d3 = tf.keras.layers.Dense(1,activation='relu')

    @tf.function
    def call(self, x):
        x = self.d2(x)
        return self.d3(x)

class VGG162D(tf.keras.Model):
    def __init__(self,inputDim = [122,146,1]):
        super(VGG162D,self).__init__()
        #tf.keras.layers.Input(shape=(224, 224, 3))
        # Block 1
        self.conv1_0 = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=(3, 3),
                               activation='relu',
                               padding='same',
                               name='block1_conv1',
                               input_shape=(122,146,1))
        self.conv1_1 = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=(3, 3),
                               activation='relu',
                               padding='same',
                               name='block1_conv2')
        self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool')
        # Block 2
        self.conv2_0 = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=(3, 3),
                               activation='relu',
                               padding='same',
                               name='block2_conv1')
        self.conv2_1 = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=(3, 3),
                               activation='relu',
                               padding='same',
                               name='block2_conv2')
        self.maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool')

        # Block 3
        self.conv3_0 = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=(3, 3),
                               activation='relu',
                               padding='same',
                               name='block3_conv1')
        self.conv3_1 = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=(3, 3),
                               activation='relu',
                               padding='same',
                               name='block3_conv2')
        self.conv3_2 = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=(3, 3),
                               activation='relu',
                               padding='same',
                               name='block3_conv3')
        self.maxpool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool')
        # Block 4
        self.conv4_0 = tf.keras.layers.Conv2D(filters=512,
                               kernel_size=(3, 3),
                               activation='relu',
                               padding='same',
                               name='block4_conv1')
        self.conv4_1 = tf.keras.layers.Conv2D(filters=512,
                               kernel_size=(3, 3),
                               activation='relu',
                               padding='same',
                               name='block4_conv2')
        self.conv4_2 = tf.keras.layers.Conv2D(filters=512,
                               kernel_size=(3, 3),
                               activation='relu',
                               padding='same',
                               name='block4_conv3')
        self.maxpool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block4_pool')
        # Block 5
        self.conv5_0 = tf.keras.layers.Conv2D(filters=512,
                               kernel_size=(3, 3),
                               activation='relu',
                               padding='same',
                               name='block5_conv1')
        self.conv5_1 = tf.keras.layers.Conv2D(filters=512,
                               kernel_size=(3, 3),
                               activation='relu',
                               padding='same',
                               name='block5_conv2')
        self.conv5_2 = tf.keras.layers.Conv2D(filters=512,
                               kernel_size=(3, 3),
                               activation='relu',
                               padding='same',
                               name='block5_conv3')
        self.maxpool5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block5_pool')
        # Block 6
        #self.flatten1 = tf.keras.layers.Flatten(name='Flatten')
        #tf.keras.layers.Dense(4096, activation='relu', name='fc1'),
        #tf.keras.layers.Dense(4096, activation='relu', name='fc2'),
        #tf.keras.layers.Dense(10, activation='softmax', name='predictions')

    @tf.function
    def call(self,x):
        x = self.conv1_0(x)
        x = self.conv1_1(x)
        x = self.maxpool1(x)

        x = self.conv2_0(x)
        x = self.conv2_1(x)
        x = self.maxpool2(x)


        x = self.conv3_0(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.maxpool3(x)

        x = self.conv4_0(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.maxpool4(x)

        x = self.conv5_0(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.maxpool5(x)

        #x = self.flatten1(x)
        return x


class VGG162DLite(tf.keras.Model):
    def __init__(self, inputDim=[122, 146, 1]):
        super(VGG162DLite, self).__init__()
        # tf.keras.layers.Input(shape=(224, 224, 3))
        # Block 1
        self.conv1_0 = tf.keras.layers.Conv2D(filters=64,
                                              kernel_size=(3, 3),
                                              activation='relu',
                                              padding='same',
                                              name='block1_conv1',
                                              input_shape=(122, 146, 1))
        self.conv1_1 = tf.keras.layers.Conv2D(filters=64,
                                              kernel_size=(3, 3),
                                              activation='relu',
                                              padding='same',
                                              name='block1_conv2')
        self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool')
        # Block 2
        self.conv2_0 = tf.keras.layers.Conv2D(filters=128,
                                              kernel_size=(3, 3),
                                              activation='relu',
                                              padding='same',
                                              name='block2_conv1')
        self.conv2_1 = tf.keras.layers.Conv2D(filters=128,
                                              kernel_size=(3, 3),
                                              activation='relu',
                                              padding='same',
                                              name='block2_conv2')
        self.maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool')

        # Block 3
        self.conv3_0 = tf.keras.layers.Conv2D(filters=256,
                                              kernel_size=(3, 3),
                                              activation='relu',
                                              padding='same',
                                              name='block3_conv1')
        self.conv3_1 = tf.keras.layers.Conv2D(filters=256,
                                              kernel_size=(3, 3),
                                              activation='relu',
                                              padding='same',
                                              name='block3_conv2')
        self.conv3_2 = tf.keras.layers.Conv2D(filters=256,
                                              kernel_size=(3, 3),
                                              activation='relu',
                                              padding='same',
                                              name='block3_conv3')
        self.maxpool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool')
        # Block 4
        self.conv4_0 = tf.keras.layers.Conv2D(filters=512,
                                              kernel_size=(3, 3),
                                              activation='relu',
                                              padding='same',
                                              name='block4_conv1')
        self.conv4_1 = tf.keras.layers.Conv2D(filters=512,
                                              kernel_size=(3, 3),
                                              activation='relu',
                                              padding='same',
                                              name='block4_conv2')
        self.conv4_2 = tf.keras.layers.Conv2D(filters=512,
                                              kernel_size=(3, 3),
                                              activation='relu',
                                              padding='same',
                                              name='block4_conv3')
        self.maxpool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block4_pool')
        # Block 5
        self.conv5_0 = tf.keras.layers.Conv2D(filters=512,
                                              kernel_size=(3, 3),
                                              activation='relu',
                                              padding='same',
                                              name='block5_conv1')
        self.conv5_1 = tf.keras.layers.Conv2D(filters=512,
                                              kernel_size=(3, 3),
                                              activation='relu',
                                              padding='same',
                                              name='block5_conv2')
        self.conv5_2 = tf.keras.layers.Conv2D(filters=512,
                                              kernel_size=(3, 3),
                                              activation='relu',
                                              padding='same',
                                              name='block5_conv3')
        self.maxpool5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block5_pool')
        # Block 6
        # self.flatten1 = tf.keras.layers.Flatten(name='Flatten')
        # tf.keras.layers.Dense(4096, activation='relu', name='fc1'),
        # tf.keras.layers.Dense(4096, activation='relu', name='fc2'),
        # tf.keras.layers.Dense(10, activation='softmax', name='predictions')

    @tf.function
    def call(self, x):
        x = self.conv1_0(x)
        x = self.conv1_1(x)
        x = self.maxpool1(x)

        x = self.conv2_0(x)
        x = self.conv2_1(x)
        x = self.maxpool2(x)

        x = self.conv3_0(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.maxpool3(x)

        x = self.conv4_0(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.maxpool4(x)

        x = self.conv5_0(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.maxpool5(x)

        # x = self.flatten1(x)
        return x


class ConvLSTMVGG16(tf.keras.Model):
    #####################
    # model Name : 사용하고자 하는 CNN Model 이름
    # numData : 2개의 배열
    #           0 번지 : 2D 데이터 개수 (fore)
    #           1 번지 : 3D 데이터 개수 (obs)
    # numLayer : Fully Connected 층 수(Wight기준)
    # inputDim : conv함수에 들어갈 input Image의 Dim데이터
    #               [Depth, Height, Width]
    # layerDim : 각 층 별 노드의 수([input, hid1, hid2, ..., output])
    # numGPU   : 사용할 GPU 개수
    # learnRate : Network에 사용할 학습률
    def __init__(self, modelName="VGG16", numData=[1, 5], numLayer=3, inputDim=[5, 122, 146],
                 layerDim=[42, 4000, 2000, 1], numGPU=1, numGPUAuto = False, learnRate=0.01, preTraining=False, fineTuning=False,
                 onlyCNN=False, batchsize = 5,hidden_dim = 15):
        super(ConvLSTMVGG16,self).__init__()
        self.batchsize = batchsize
        self.modelName = copy.deepcopy(modelName)
        self.numdata = copy.deepcopy(numData)
        self.numLayer = copy.deepcopy(numLayer)
        self.inputDim = copy.deepcopy(inputDim)
        self.layerDim = copy.deepcopy(layerDim)

        self.learnRate = copy.deepcopy(learnRate)
        self.preTraining = copy.deepcopy(preTraining)
        self.fineTuning = copy.deepcopy(fineTuning)
        self.onlyCNN = copy.deepcopy(onlyCNN)
        if self.fineTuning == True:
            self.trainable = False
        else:
            self.trainable = True
        print(self.trainable)

        self.hidden_dim = hidden_dim
        self.convLSTMEncoderDepth = 3
        self.convLSTMDecoderDepth = 3
        #self.convLSTMEncoderDepth = 1#default depth
        #self.convLSTMDecoderDepth = 1#default depth

        self.dnnlayer = [DNNpredict(self.layerDim) for i in range(10)]
        self.outputConvLSTM = [0 for i in range(10)]#20210902. T6~T15까지의 convlstm output저장
        #self.pre_vgg_model = [VGG162D(self.inputDim) for i in range(10)]
        self.pre_vgg_model = VGG162D(self.inputDim)
        self.outputVGG3DFromLSTM = [0 for i in range(10)]#20210902. T6~T15까지의 convlstm output저장

    @tf.function
    def call(self,inputs_cnn,inputs_dnn,inputs_ecell,inputs_ehidden,inputs_dcell,inputs_dhidden):
        #Model(inputs=[visible1, visible2], outputs=output)
        #convLSTM으로 선언된 모델에서 아래와 같이 CNN, DNN입력이 둘다 들어야됨.
        #그러기 위해서 위와 같이 input에 리스트로 전달을 해줘야 된다.
        #output일때도 아래와 같이 list로 하여 여러개의 출력이 나올 수 있도록 설정한다.
        #self.inputdim = [5,122,146]#depth,height,width
        #inputs_cnn = tf.keras.Input(shape=(self.inputDim[0], self.inputDim[1] * self.inputDim[2],self.inputDim[0],))

        #inputs_dnn = tf.keras.Input(shape=(self.layerDim[0],))#tf.placeholder(tf.float32, [None, self.layerDim[0]])
        #inputs_ecell = tf.keras.Input(shape=(self.inputDim[1],self.inputDim[2],self.hidden_dim,))
        #inputs_ehidden = tf.keras.Input(shape=(self.inputDim[1],self.inputDim[2],self.hidden_dim,))
        #inputs_dcell = tf.keras.Input(shape=(self.inputDim[1],self.inputDim[2],self.hidden_dim,))
        #inputs_dhidden = tf.keras.Input(shape=(self.inputDim[1],self.inputDim[2],self.hidden_dim,))

        self.convLSTMBlock = Seq2Seq(ecell=inputs_ecell,ehidden=inputs_ehidden,dcell=inputs_dcell,dhidden=inputs_dhidden,hidden = self.hidden_dim,enc_num_layers= self.convLSTMEncoderDepth,
                                     dec_num_layers=self.convLSTMDecoderDepth,future_len=10)  # 20210921추가, convlstm블락

        inputclstmimage = tf.reshape(inputs_cnn,[-1, self.inputDim[0], self.inputDim[1], self.inputDim[2],self.inputDim[0]])  # self.inputDim[0] : T1~T5)
        conv_hidden_output = self.convLSTMBlock(inputs_ecell,inputs_ehidden,inputs_dcell,inputs_dhidden,inputclstmimage)
        convlstm_vgg_output = []
        for i in range(10):
            #self.pre_vgg_model[i] = VGG162D(self.inputDim)
            #vgg_output_per_time = self.pre_vgg_model[i](conv_hidden_output[:, i])
            vgg_output_per_time = self.pre_vgg_model(conv_hidden_output[:, i])
            vgg_shape = vgg_output_per_time.get_shape()
            reshapesize = int(vgg_shape[1])*int(vgg_shape[2])*int(vgg_shape[3])
            flatten_vgg_output = tf.reshape(vgg_output_per_time,[-1,reshapesize])
            convlstm_vgg_output.append(flatten_vgg_output)
        Tn_DNNData = [tf.concat([convlstm_vgg_output[i], inputs_dnn[:, 14 + i * 16:14 + (i + 1) * 16], inputs_dnn[:, 174:186]],1) for i in range(10)]
        dnn_output = [0 for i in range(10)]

        for i in range(10):
            dnn_output[i] = self.dnnlayer[i](Tn_DNNData[i])
        return tf.concat(dnn_output,1)
        #return tf.keras.Model([inputs_cnn,inputs_dnn,inputs_ecell,inputs_ehidden,inputs_dcell,inputs_dhidden],dnn_output)


class ConvLSTMVGG16_fixed(tf.keras.Model):
    #####################
    # model Name : 사용하고자 하는 CNN Model 이름
    # numData : 2개의 배열
    #           0 번지 : 2D 데이터 개수 (fore)
    #           1 번지 : 3D 데이터 개수 (obs)
    # numLayer : Fully Connected 층 수(Wight기준)
    # inputDim : conv함수에 들어갈 input Image의 Dim데이터
    #               [Depth, Height, Width]
    # layerDim : 각 층 별 노드의 수([input, hid1, hid2, ..., output])
    # numGPU   : 사용할 GPU 개수
    # learnRate : Network에 사용할 학습률
    def __init__(self, modelName="VGG16", numData=[1, 5], numLayer=3, inputDim=[5, 122, 146],
                 layerDim=[42, 4000, 2000, 1], numGPU=1, numGPUAuto = False, learnRate=0.01, preTraining=False, fineTuning=False,
                 onlyCNN=False, batchsize = 5,hidden_dim = 15):
        super(ConvLSTMVGG16_fixed,self).__init__()
        self.batchsize = batchsize
        self.modelName = copy.deepcopy(modelName)
        self.numdata = copy.deepcopy(numData)
        self.numLayer = copy.deepcopy(numLayer)
        self.inputDim = copy.deepcopy(inputDim)
        self.layerDim = copy.deepcopy(layerDim)

        self.learnRate = copy.deepcopy(learnRate)
        self.preTraining = copy.deepcopy(preTraining)
        self.fineTuning = copy.deepcopy(fineTuning)
        self.onlyCNN = copy.deepcopy(onlyCNN)
        if self.fineTuning == True:
            self.trainable = False
        else:
            self.trainable = True
        print(self.trainable)

        self.hidden_dim = hidden_dim
        self.convLSTMEncoderDepth = 3
        self.convLSTMDecoderDepth = 3
        #self.convLSTMEncoderDepth = 1#default depth
        #self.convLSTMDecoderDepth = 1#default depth
        self.convLSTMBlock = Seq2Seq(hidden=self.hidden_dim,
                                     enc_num_layers=self.convLSTMEncoderDepth,
                                     dec_num_layers=self.convLSTMDecoderDepth, future_len=10)  # 20210921추가, convlstm블락

        self.dnnlayer = [DNNpredict(self.layerDim) for i in range(10)]
        self.outputConvLSTM = [0 for i in range(10)]#20210902. T6~T15까지의 convlstm output저장
        #self.pre_vgg_model = [VGG162D(self.inputDim) for i in range(10)]
        self.pre_vgg_model = VGG162D(self.inputDim)
        self.outputVGG3DFromLSTM = [0 for i in range(10)]#20210902. T6~T15까지의 convlstm output저장

    @tf.function
    def call(self,inputs_cnn,inputs_dnn,inputs_ecell,inputs_ehidden,inputs_dcell,inputs_dhidden):
        inputclstmimage = tf.reshape(inputs_cnn,[-1, self.inputDim[0], self.inputDim[1], self.inputDim[2],self.inputDim[0]])  # self.inputDim[0] : T1~T5)
        conv_hidden_output = self.convLSTMBlock(inputclstmimage)
        convlstm_vgg_output = []
        for i in range(10):
            #self.pre_vgg_model[i] = VGG162D(self.inputDim)
            #vgg_output_per_time = self.pre_vgg_model[i](conv_hidden_output[:, i])
            vgg_output_per_time = self.pre_vgg_model(conv_hidden_output[:, i])
            vgg_shape = vgg_output_per_time.get_shape()
            reshapesize = int(vgg_shape[1])*int(vgg_shape[2])*int(vgg_shape[3])
            flatten_vgg_output = tf.reshape(vgg_output_per_time,[-1,reshapesize])
            convlstm_vgg_output.append(flatten_vgg_output)
        Tn_DNNData = [tf.concat([convlstm_vgg_output[i], inputs_dnn[:, 14 + i * 16:14 + (i + 1) * 16], inputs_dnn[:, 174:186]],1) for i in range(10)]
        dnn_output = [0 for i in range(10)]

        for i in range(10):
            dnn_output[i] = self.dnnlayer[i](Tn_DNNData[i])
        return tf.concat(dnn_output,1)
        #return tf.keras.Model([inputs_cnn,inputs_dnn,inputs_ecell,inputs_ehidden,inputs_dcell,inputs_dhidden],dnn_output)
