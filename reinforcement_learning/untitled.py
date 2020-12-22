        # Create a random folder
        random_foldername = [np.random.choice(string.ascii_letters) 
                           for _ in range(12)]

        random_foldername = ''.join(random_foldername)
        folder_location = os.path.join('rl_sample_images', random_foldername)
        os.makedir(folder_location)

        # Put the image files in that folder

        # create a mapping from folders to filenames
        images = [img1, img2, img_mid]
        images += eight_images

        filenames = ['img1', 'img2', 'img_mid']
        filenames += ['sample_{}'.format(i) for i in range(1,9)]

        filepaths = [os.path.join(folder_location, filename)
                     for filename in filenames]

        mapping = [(images[i], filepaths[i]) for i in range(len(images))]
        
        _ = [scipy.misc.imsave(filepath, image) 
             for (image, filepath) in mapping]

        image_listing = dict(zip(filenames, filepaths))
        image_listing['random_folder'] = random_foldername
        level_mapping = {0:0.3, 1:0.5, 2:0.7}
        image_listing['level'] = level_mapping[int_lvl]*100

        path_pipe.put(image_listing, block=True)