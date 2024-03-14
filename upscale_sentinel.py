import numpy as np
import rasterio
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
#from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers.legacy import SGD, Adam
from keras.applications.vgg16 import VGG16
import keras.backend as K
import tensorflow as tf
from tqdm.notebook import tqdm

class SentinelDataProcessor:
    def __init__(self, path: str, section_size=256, upscaling_factor=4, test=False):
        self.path = path
        self.section_size = section_size
        self.scaling_factor = 1/upscaling_factor
        #self.channels = [3, 2, 1] #R: 3, G: 2, B: 1 according to Sentinel 2 multispectral channel order
        self.hr_images = self.generate_rgb_sections(test=test)
        self.lr_images = self.downsize_images()
        self.input_train, self.input_eval, self.output_train, self.output_eval = train_test_split(self.lr_images, self.hr_images, test_size=0.2, random_state=42)
        self.input_test, self.input_val, self.output_test, self.output_val = train_test_split(self.input_eval, self.output_eval, test_size=0.2, random_state=42)
    
    def generate_rgb_sections(self, test=False):
        sections = []
        file_list = sorted(os.listdir(self.path))

        for filename in file_list:
            if filename.endswith(".tif"):
                file_path = os.path.join(self.path, filename)
                print(filename)
                with rasterio.open(file_path) as src:
                    if src.count == 3:  # Check if the file has 12 channels
                        # Process and store sections in chunks to save memory
                        for i in range(0, src.height, self.section_size):
                            for j in range(0, src.width, self.section_size):
                                if (i + self.section_size <= src.height) and (j + self.section_size <= src.width):
                                    # Read only the necessary slice to reduce memory usage
                                    data = src.read(window=rasterio.windows.Window(j, i, self.section_size, self.section_size))
                                    rgb_data = data[[0, 1, 2], :, :]  # Adjust channels here, if needed
                                    rgb_data = np.clip(rgb_data, 0, 1)
                                    if np.any(np.logical_or(rgb_data == 0, np.isnan(rgb_data))):
                                        continue
                                    section = rgb_data.transpose((1, 2, 0))
                                    sections.append(section)
                if test:
                    break

        sections = np.stack(sections)
        print(f'Image data array shape: {sections.shape}')
        return sections
    
    def downsize_images(self):
        new_height = int(self.hr_images.shape[1] * self.scaling_factor)
        new_width = int(self.hr_images.shape[2] * self.scaling_factor)

        # Initialize the downsized images array with a more memory-efficient dtype
        downsized_images = np.empty((self.hr_images.shape[0], new_height, new_width, 3))
        
        print('downsampling input data...')

        for i in tqdm(range(self.hr_images.shape[0])):
            # Resize images one by one to save memory, consider batch processing if possible
            downsized_images[i] = cv2.resize(self.hr_images[i], (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        downsized_images = np.clip(downsized_images, 0, 1)

        return downsized_images
    
    def plot_samples(self, num_samples=3, idx=None):
        # Ensure there are enough samples to plot
        num_samples = min(num_samples, self.lr_images.shape[0])
        
        if idx is None:
            # Generate random, unique indices for the samples
            indices = np.random.choice(self.lr_images.shape[0], size=num_samples, replace=False)
        else:
            # If specific indices are provided, use them directly
            indices = np.array([idx] * num_samples) if np.isscalar(idx) else np.array(idx)
            # Ensure we don't exceed the number of available samples
            indices = indices[:num_samples]
        
        plt.close('all')
        plt.figure(figsize=(12, num_samples*5))
        
        for i, index in enumerate(indices):
            # Plot the low resolution image
            plt.subplot(num_samples, 2, i*2 + 1)
            plt.imshow(self.lr_images[index, :, :, :])
            plt.title(f'Low Res - Sample {index}')
            plt.axis('off')
            
            # Plot the corresponding high resolution image
            plt.subplot(num_samples, 2, i*2 + 2)
            plt.imshow(self.hr_images[index, :, :, :])
            plt.title(f'High Res - Sample {index}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()



        

class UpscaleModelTrainer:
    def __init__(self, input_shape, filter_gen=64, upscale_factor=2):
        self.input_shape = input_shape
        self.upscale_factor = upscale_factor
        self.vgg = self.build_vgg()
        self.loss_weights = [1, 2e-1, 1e-3, 1e-8] # mse_loss, vgg_loss, adv_loss, tv_loss
        self.generator = self.build_generator(filters=filter_gen)
        self.discriminator = self.build_discriminator()
        self.model = self.define_srgan()
        self.upscalegen = self.setup_datagen()
        #self.callbacks = self.define_callbacks()
        self.history = None
        
    def setup_datagen(self):
        # datagen = ImageDataGenerator(
        #     rotation_range=45, 
        #     width_shift_range=0.1, 
        #     height_shift_range=0.1,
        #     horizontal_flip=True, 
        #     vertical_flip=True, 
        #     fill_mode='reflect')
        # labelgen = ImageDataGenerator(
        #     rotation_range=45, 
        #     width_shift_range=0.1, 
        #     height_shift_range=0.1,
        #     horizontal_flip=True, 
        #     vertical_flip=True, 
        #     fill_mode='reflect')
        upscalegen = ImageDataGenerator(
            # rotation_range=45, 
            # width_shift_range=0.1, 
            # height_shift_range=0.1,
            # fill_mode='reflect',
            horizontal_flip=True, 
            vertical_flip=True)
        return upscalegen

    ### model blacks
    def residual_block(self, x, num_filter):
        skip = x
        x = layers.Conv2D(num_filter, kernel_size=3, kernel_initializer='he_uniform', padding='same')(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        x = layers.PReLU(shared_axes=[1, 2])(x)  # shared_axes argument makes PReLU channel-wise
        x = layers.Conv2D(num_filter, kernel_size=3, kernel_initializer='he_uniform', padding='same')(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        x = layers.Add()([x, skip])
        return x

    def upsample_block(self, x, num_filters, method='pixelshuffle'):
        if method == 'pixelshuffle':
            x = layers.Conv2D(num_filters*4, kernel_size=3, kernel_initializer='he_uniform', padding='same')(x)
            x = layers.Lambda(lambda y: tf.nn.depth_to_space(y, 2))(x)  # This acts as PixelShuffler
            x = layers.PReLU(shared_axes=[1, 2])(x)
        if method == 'resize':
            x = tf.image.resize(x, (x.shape[1]*2, x.shape[2]*2), method='nearest')
            x = layers.Conv2D(num_filters, kernel_size=3, kernel_initializer='he_uniform', padding='same')(x)
            x = layers.PReLU(shared_axes=[1, 2])(x)
        return x
    
    ### SRGAN components
    def build_generator(self, filters=64, tvloss=True):
        inputs = layers.Input(shape=self.input_shape)
        x = layers.Conv2D(filters, kernel_size=5, kernel_initializer='he_uniform', padding='same')(inputs)
        x = layers.PReLU(shared_axes=[1, 2])(x)
        skip_connection = x

        for _ in range(5):  # Number of residual blocks
            x = self.residual_block(x, num_filter=filters)

        x = layers.Conv2D(filters, kernel_size=3, kernel_initializer='he_uniform', padding='same')(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        x = layers.Add()([x, skip_connection])

        for _ in range(self.upscale_factor//2):
            x = self.upsample_block(x, filters*4, method='resize')

        outputs = layers.Conv2D(3, kernel_size=5, padding='same', activation='tanh')(x)
        if tvloss:
            outputs = TVLossLayer(weight=self.loss_weights[3])(outputs)

        model = models.Model(inputs=inputs, outputs=outputs, name='gen')
        model.summary()
        return model
    
    def build_discriminator(self, conv_blocks=[(8, 2), (16, 1), (16, 2), (32, 1), (32, 2)], dense_units=8):
        inputs = layers.Input(shape=(self.input_shape[0]*self.upscale_factor, self.input_shape[1]*self.upscale_factor, self.input_shape[2]))

        x = inputs
        x = layers.Conv2D(8, kernel_size=3, kernel_initializer='he_uniform', strides=1, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        for filters, stride in conv_blocks:
            x = layers.Conv2D(filters, kernel_size=3, kernel_initializer='he_uniform', strides=stride, padding='same')(x)
            x = layers.BatchNormalization(momentum=0.8)(x)
            x = layers.LeakyReLU(alpha=0.02)(x)
            x = layers.Dropout(0.2)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(dense_units)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(inputs=inputs, outputs=outputs, name='discriminator')
        model.summary()
        return model
    
    def build_vgg(self):

        input_layer = layers.Input(shape=(self.input_shape[0]*self.upscale_factor, self.input_shape[1]*self.upscale_factor, self.input_shape[2]))
        
        # Replicate the grayscale channel to mimic RGB channels
        #x = layers.Lambda(lambda x: tf.tile(x, [1, 1, 1, 3]))(input_layer)
        
        vgg = VGG16(include_top=False, weights='imagenet', input_tensor=input_layer)
        vgg.trainable = False
        for layer in vgg.layers:
            layer.trainable = False
        # Adjusted to use the last convolutional layer of VGG16
        model = models.Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv3').output)
        model.trainable = False
        return model
    
    def define_srgan(self):
        #disc_optimizer = SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
        disc_optimizer = Adam(learning_rate=0.00003, beta_1=0.5, beta_2=0.999)
        self.discriminator.compile(optimizer=disc_optimizer, loss='binary_crossentropy', metrics=['accuracy'])  # Compile the discriminator
        self.discriminator.trainable = False  # Make discriminator non-trainable
        input_low_res = layers.Input(shape=self.input_shape)
        fake_high_res = self.generator(input_low_res)
        validity = self.discriminator(fake_high_res)
        vgg_features = self.vgg(fake_high_res)
        srgan = models.Model(input_low_res, [fake_high_res, vgg_features, validity])
        #optimizer = SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
        optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
        srgan.compile(optimizer=optimizer,
                        loss=["mse", "mse", "binary_crossentropy"],
                        loss_weights = self.loss_weights[:3])

        return srgan
    
    def train_upscale(self, training_data, validation_data, epochs=50, batch_size=16, aug_data=True):

        if self.history == None:
            self.history = {'d_real_loss': [], 
                            'd_fake_loss': [], 
                            'd_real_acc': [], 
                            'd_fake_acc': [], 
                            'g_loss': [], 
                            'g_mse': [],
                            'g_vgg': [],
                            'g_adv': [],
                            'tv_loss': [],
                            'val_d_real_loss': [], 
                            'val_d_fake_loss': [], 
                            'val_d_real_acc': [], 
                            'val_d_fake_acc': [], 
                            'val_g_loss': [],
                            'val_g_mse': [],
                            'val_g_vgg': [],
                            'val_g_adv': [],
                            'val_tv_loss': []}
            
        lr_images, hr_images = training_data
        val_lr_images, val_hr_images = validation_data
        
        #datagen_input, _ = self.setup_datagen()
        indices = np.arange(lr_images.shape[0])
        d_loss_acc = np.zeros(4)
        d_loss = np.ones(2)
        g_loss = np.ones(4)
        g_loss_acc = np.ones(4)
        
        disc_train = True
        
        for epoch in tqdm(range(epochs), desc='epochs', position=0):
            np.random.shuffle(indices)  # Shuffle at the start of each epoch
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))
            
            for i in tqdm(range(0, lr_images.shape[0], batch_size), desc='batches', position=1, leave=False):
                batch_indices = indices[i:i+batch_size]

                if len(batch_indices) < batch_size: continue  # Skip this batch

                imgs_lr = lr_images[batch_indices]
                imgs_hr = hr_images[batch_indices]

                seed = np.random.randint(0, 99)

                if aug_data:
                    augmented_images_lr = next(self.upscalegen.flow(imgs_lr, batch_size=batch_size, shuffle=False, seed=seed))
                    augmented_images_hr = next(self.upscalegen.flow(imgs_hr, batch_size=batch_size, shuffle=False, seed=seed))
                else:
                    augmented_images_lr = imgs_lr
                    augmented_images_hr = imgs_hr

                gen_imgs = self.generator.predict(augmented_images_lr, verbose=0)
                
                if disc_train:
                    self.discriminator.trainable = True
                else:
                    self.discriminator.trainable = False
                d_loss_real = self.discriminator.train_on_batch(augmented_images_hr, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                #d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                self.discriminator.trainable = False  # Only needs to be set once per batch
           
                vgg_features = self.vgg(augmented_images_hr)

                # Train the generator
                g_loss = self.model.train_on_batch(augmented_images_lr, [augmented_images_hr, vgg_features, valid])

                d_loss_acc += np.array([d_loss_real[0], d_loss_real[1], d_loss_fake[0], d_loss_fake[1]])
                g_loss_acc += np.array([g_loss[0], g_loss[1], g_loss[2], g_loss[3]])

                tv_loss = g_loss[0] - self.loss_weights[0]*g_loss[1] - self.loss_weights[1]*g_loss[2] - self.loss_weights[2]*g_loss[3]

                # Logging
                print(f'\rEpoch {epoch+1}/{epochs} \t[Disc loss: (r {(d_loss_real[0]):.6f}/f {(d_loss_fake[0]):.6f}), acc.: (r {d_loss_real[1]:.4f}/f {d_loss_fake[1]:.4f})] [Gen loss: {g_loss[0]:.6f}, mse: {self.loss_weights[0]*g_loss[1]:.6f}, vgg: {self.loss_weights[1]*g_loss[2]:.6f}, adv: {self.loss_weights[2]*g_loss[3]:.6f}, tv: {tv_loss:.6f}]', end='')

            d_loss_acc /= (lr_images.shape[0]//batch_size+1)
            g_loss_acc /= (lr_images.shape[0]//batch_size+1)

            gmse = self.loss_weights[0]*g_loss_acc[1]
            gvgg = self.loss_weights[1]*g_loss_acc[2]
            gadv = self.loss_weights[2]*g_loss_acc[3]
            tv_loss = g_loss_acc[0] - gmse - gvgg - gadv

            self.history['d_real_loss'].append(d_loss_acc[0])
            self.history['d_fake_loss'].append(d_loss_acc[2])
            self.history['d_real_acc'].append(d_loss_acc[1])
            self.history['d_fake_acc'].append(d_loss_acc[3]) 
            self.history['g_loss'].append(g_loss_acc[0])
            self.history['g_mse'].append(gmse)
            self.history['g_vgg'].append(gvgg)
            self.history['g_adv'].append(gadv)
            self.history['tv_loss'].append(tv_loss)
            
            if d_loss_acc[3] > 0.9975 or d_loss_acc[1] > 0.9975:
                disc_train = False
            else:
                disc_train = True

            print(f"\rEpoch {epoch+1}/{epochs} \t[Disc loss: (r {(d_loss_acc[0]):.6f}/f {(d_loss_acc[2]):.6f}), acc.: (r {d_loss_acc[1]:.4f}/f {d_loss_acc[3]:.4f})] [Gen loss: {g_loss_acc[0]:.6f}, mse: {gmse:.6f}, vgg: {gvgg:.6f}, adv: {gadv:.6f}, tv: {tv_loss:.6f}]")

            val_d_loss_real_accum = []
            val_d_loss_fake_accum = []
            val_g_loss_accum = []
    
            # Process validation data in batches
            for i in tqdm(range(0, val_lr_images.shape[0], batch_size), desc='Validation', position=1, leave=False):
                batch_indices = np.arange(i, min(i+batch_size, val_lr_images.shape[0]))
        
                if len(batch_indices) < batch_size:  # Ensure full batch size, otherwise skip
                    continue
        
                batch_val_lr_images = val_lr_images[batch_indices]
                batch_val_hr_images = val_hr_images[batch_indices]
        
                val_gen_imgs = self.generator.predict(batch_val_lr_images, verbose=0)
        
                # Evaluate the discriminator on batch validation data
                val_d_loss_real = self.discriminator.evaluate(batch_val_hr_images, np.ones((len(batch_indices), 1)), verbose=0)
                val_d_loss_fake = self.discriminator.evaluate(val_gen_imgs, np.zeros((len(batch_indices), 1)), verbose=0)
        
                val_vgg_features = self.vgg(batch_val_hr_images)
        
                # Evaluate the generator on batch validation data
                val_g_loss = self.model.evaluate(batch_val_lr_images, [batch_val_hr_images, val_vgg_features, np.ones((len(batch_indices), 1))], verbose=0)
        
                # Accumulate the validation losses
                val_d_loss_real_accum.append(val_d_loss_real)
                val_d_loss_fake_accum.append(val_d_loss_fake)
                val_g_loss_accum.append(val_g_loss)

            # Calculate the average of the accumulated validation losses and accuracies
            val_d_real_loss_avg = np.mean([loss[0] for loss in val_d_loss_real_accum])
            val_d_real_acc_avg = np.mean([acc[1] for acc in val_d_loss_real_accum])
            val_d_fake_loss_avg = np.mean([loss[0] for loss in val_d_loss_fake_accum])
            val_d_fake_acc_avg = np.mean([acc[1] for acc in val_d_loss_fake_accum])
            val_g_loss_avg = np.mean([loss[0] for loss in val_g_loss_accum], axis=0)
            val_g_mse_avg, val_g_vgg_avg, val_g_adv_avg = [np.mean([loss[i+1] for loss in val_g_loss_accum]) for i in range(3)]
            val_tv_loss_avg = val_g_loss_avg - self.loss_weights[0]*val_g_mse_avg - self.loss_weights[1]*val_g_vgg_avg - self.loss_weights[2]*val_g_adv_avg
        
            # Update history with averaged validation data
            self.history['val_d_real_loss'].append(val_d_real_loss_avg)
            self.history['val_d_fake_loss'].append(val_d_fake_loss_avg)
            self.history['val_d_real_acc'].append(val_d_real_acc_avg)
            self.history['val_d_fake_acc'].append(val_d_fake_acc_avg)
            self.history['val_g_loss'].append(val_g_loss_avg)
            self.history['val_g_mse'].append(self.loss_weights[0]*val_g_mse_avg)
            self.history['val_g_vgg'].append(self.loss_weights[1]*val_g_vgg_avg)
            self.history['val_g_adv'].append(self.loss_weights[2]*val_g_adv_avg)
            self.history['val_tv_loss'].append(val_tv_loss_avg)

            #print(f"Validation: \t[Disc loss: {val_d_loss[0]:.6f}, acc.: (r {val_d_loss_real[1]:.4f}/f {val_d_loss_fake[1]:.4f})]                [Gen loss: {val_g_loss[0]:.6f}, mse: {self.loss_weights[0]*val_g_loss[1]:.6f}, vgg: {self.loss_weights[1]*val_g_loss[2]:.6f}, adv: {self.loss_weights[2]*val_g_loss[3]:.6f}, tv: {val_tv_loss:.6f}]")
            print(f"Validation: \t"
                  f"[Disc loss: (r {val_d_real_loss_avg:.6f}/f {val_d_fake_loss_avg:.6f}), acc.: (r {val_d_real_acc_avg:.4f}/f {val_d_fake_acc_avg:.4f})] "
                  f"[Gen loss: {val_g_loss_avg:.6f}, mse: {self.loss_weights[0]*val_g_mse_avg:.6f}, vgg: {self.loss_weights[1]*val_g_vgg_avg:.6f}, adv: {self.loss_weights[2]*val_g_adv_avg:.6f}, tv: {val_tv_loss_avg:.6f}]")
            print('-' * 120)
            # Save images at intervals
            #if (epoch+1) % 2 == 0:
            self.save_temp_images(epoch, val_gen_imgs)

    def save_temp_images(self, epoch, generated_images):

        generated_images = np.clip(generated_images, 0, 1)
        idxs = np.arange(16)

        plt.close('all')
        plt.figure(figsize=(10, 10))
        for i in range(16):
            plt.subplot(4, 4, i+1)
            plt.imshow(generated_images[idxs[i], :, :, :])#, cmap='magma', vmin=-1, vmax=1)
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'/content/drive/My Drive/Colab Notebooks/srgan_images/images_at_epoch_{epoch}.png')
        plt.close()
    
    def load_model(self, model_path):
        self.model = models.load_model(model_path, safe_mode=False)

    def predict_upscale_data(self, data):
        return self.model.predict(data)
    
    def plot_history(self):

        #keys = ['d_real_loss', 'd_fake_loss', 'd_real_acc', 'd_fake_acc', 'g_loss', 'val_d_real_loss', 'val_d_fake_loss', 'val_d_real_acc', 'val_d_fake_acc', 'val_g_loss']

        plt.close('all')
        # Plot training & validation accuracy values
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1)
        plt.plot(self.history['d_real_acc'], label='d_real_acc', color='C0', alpha=0.5)
        plt.plot(self.history['d_fake_acc'], label='d_fake_acc', color='C1', alpha=0.5)
        plt.plot(self.history['val_d_real_acc'], label='val_d_real_acc', color='C0')
        plt.plot(self.history['val_d_fake_acc'], label='val_d_fake_acc', color='C1')
        
        #plt.yticks(np.arange(0.8,1.01,0.01))
        #plt.ylim(0.89,1)
        plt.title('Discriminator Accuracy')
        plt.xlabel('Epoch')
        plt.ylim(top=1.05, bottom=0)
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()

        # Plot training & validation loss values
        plt.subplot(2, 2, 3)
        plt.plot(self.history['d_real_loss'], label='d_real_loss', color='C0', alpha=0.5)
        plt.plot(self.history['d_fake_loss'], label='d_fake_loss', color='C1', alpha=0.5)
        plt.plot(self.history['val_d_real_loss'], label='val_d_real_loss', color='C0')
        plt.plot(self.history['val_d_fake_loss'], label='val_d_fake_loss', color='C1')
        plt.title('Discriminator Loss')
        plt.xlabel('Epoch')
        #plt.ylim(bottom=0)
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(self.history['g_loss'], label='g_loss', color='C0')
        plt.plot(self.history['val_g_loss'], label='val_g_loss', alpha=0.5, color='C0')
        plt.plot(self.history['g_mse'], label='g_mse', color='C1')
        plt.plot(self.history['val_g_mse'], label='val_g_mse', alpha=0.5, color='C1')
        plt.plot(self.history['g_vgg'], label='g_vgg', color='C2')
        plt.plot(self.history['val_g_vgg'], label='val_g_vgg', alpha=0.5, color='C2')
        plt.plot(self.history['g_adv'], label='g_adv', color='C3')
        plt.plot(self.history['val_g_adv'], label='val_g_adv', alpha=0.5, color='C3')
        plt.plot(self.history['tv_loss'], label='tv_loss', color='C4')
        plt.plot(self.history['val_tv_loss'], label='val_tv_loss', alpha=0.5, color='C4')

        plt.title('Generator Loss')
        plt.xlabel('Epoch')
        plt.ylim(bottom=0)
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()

    def visualize_upscale_pred(self, data, predictions, true_labels, num_samples=6):
    
        rand_offset = np.random.randint(0, data.shape[0] - num_samples - 1)
        print(f"Random offset: {rand_offset}, Total samples: {data.shape[0]}")

        plt.close('all')
        plt.figure(figsize=(num_samples*2, 8))

        for i in range(num_samples):
            # Original image (input)
            plt.subplot(4, num_samples, i + 1) 
            plt.imshow(data[i + rand_offset])#, cmap='magma', vmin=-1, vmax=1)
            plt.title(f'Low Res Image {i}', fontsize=8)
            plt.axis('off')

            # Bicubic upscaled image
            orig_size = data[i + rand_offset].shape[:2] 
            upscaled_size = (orig_size[1] * self.upscale_factor, orig_size[0] * self.upscale_factor)
            upscaled_image = cv2.resize(data[i + rand_offset], upscaled_size, interpolation=cv2.INTER_CUBIC)

            plt.subplot(4, num_samples, i + 1 + num_samples)
            plt.imshow(upscaled_image)#, cmap='magma', vmin=-1, vmax=1)
            plt.title(f'Bicubic Upscale {i}', fontsize=8)
            plt.axis('off')

            # Predicted mask (output)
            plt.subplot(4, num_samples, i + 1 + 2 * num_samples)
            plt.imshow(predictions[i + rand_offset])#, cmap='magma', vmin=-1, vmax=1)
            plt.title(f'Predicted Image {i}', fontsize=8)
            plt.axis('off')

            # True mask
            plt.subplot(4, num_samples, i + 1 + 3 * num_samples) 
            plt.imshow(true_labels[i + rand_offset])#, cmap='magma', vmin=-1, vmax=1)
            plt.title(f'High Res Image {i}', fontsize=8)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def update_loss_weights(self, new_weights=None, lr=0.0002):
        if new_weights is not None:
            self.loss_weights = new_weights
        
        self.model.compile(optimizer=Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999),
                                loss=["mse", "mse", "binary_crossentropy"],
                                loss_weights = self.loss_weights[:3])

    def save_generator(self, path_name: str):
        generator_without_tvloss = self.build_generator(tvloss=False)
        generator_without_tvloss.set_weights(self.generator.get_weights())
        generator_without_tvloss.save_weights(path_name)

    def save_discriminator(self, path_name: str):
        self.discriminator.save_weights(path_name)

    def load_generator(self, gen_path: str):
        self.generator.load_weights(gen_path)

    def load_discriminator(self, disc_path: str):
        self.discriminator.load_weights(disc_path)

class TVLossLayer(tf.keras.layers.Layer):
    def __init__(self, weight=3e-7, **kwargs):
        super(TVLossLayer, self).__init__(**kwargs)
        self.weight = weight  # Dynamic weight that can be updated

    def call(self, inputs):
        tv_loss = tf.reduce_sum(tf.image.total_variation(inputs)) * self.weight
        self.add_loss(tv_loss)
        return inputs

    def update_weight(self, new_weight):
        self.weight = new_weight
