from pathlib import Path

import keras
from keras.datasets import cifar10, cifar100, mnist
from keras.utils import to_categorical  # Does One-hot-encoding
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..utils.data import consolidate_bins, crop_center


def load_data(load_cat=False, **config):
    if config["dataset"] == "mnist":
        (x_train, y_train), (x_test, y_test) = load_mnist(**config)

    elif config["dataset"] == "cifar10":
        (x_train, y_train), (x_test, y_test) = load_cifar10(**config)

    elif config["dataset"] == "cifar100":
        (x_train, y_train), (x_test, y_test) = load_cifar100(**config)

    elif "sdss" in config["dataset"]:
        if load_cat:

            (
                (x_train, y_train, vals_train, z_spec_train, cat_train),
                (x_dev, y_dev, vals_dev, z_spec_dev, cat_dev),
                (x_test, y_test, vals_test, z_spec_test, cat_test),
            ) = load_sdss(load_cat=load_cat, **config)
            return (
                (x_train, y_train, vals_train, z_spec_train, cat_train),
                (x_dev, y_dev, vals_dev, z_spec_dev, cat_dev),
                (x_test, y_test, vals_test, z_spec_test, cat_test),
            )
        else:

            (
                (x_train, y_train, vals_train, z_spec_train),
                (x_dev, y_dev, vals_dev, z_spec_dev),
                (x_test, y_test, vals_test, z_spec_test),
            ) = load_sdss(load_cat=load_cat, **config)
            return (
                (x_train, y_train, vals_train, z_spec_train),
                (x_dev, y_dev, vals_dev, z_spec_dev),
                (x_test, y_test, vals_test, z_spec_test),
            )
    
    elif "satellite" in config["dataset"]:
        if config['dataloader'] == 'GOES':
            data = load_GOES(**config)
        elif config['dataloader'] == 'RMW':
            data = load_RMW(**config)
        return data
    
    else:
        raise ValueError(
            f"`{config['dataset']}` is not one of the valid datasets: "
            "'mnist', 'cifar10', 'sdss', or 'satellite'."
        )

    return (x_train, y_train), (x_test, y_test)


def load_mnist(num_class, **params):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    y_train = to_categorical(y_train.astype("float32"), num_class)
    y_test = to_categorical(y_test.astype("float32"), num_class)
    return (x_train, y_train), (x_test, y_test)


def load_cifar10(num_class, **params):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train, num_class)
    y_test = to_categorical(y_test, num_class)
    return (x_train, y_train), (x_test, y_test)


def load_cifar100(num_class, **params):
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine")
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train, num_class)
    y_test = to_categorical(y_test, num_class)
    return (x_train, y_train), (x_test, y_test)


def load_sdss(
    num_class,
    path_data,
    frac_train=0.8,
    frac_dev=0.1,
    random_state=200,
    image_scale=10.0,
    load_cat=False,
    **params,
):
    filename = f"{params['dataset']}.npz"
    paths = [
        Path(path_data),
        Path("/bgfs/jnewman/bid13/photoZ/data/pasquet2019"),
        Path("/Users/andrews/projects/photoz/data/pasquet2019"),
        Path("/home/biprateep/Documents/photozCapsNet/photozCapsNet"),
    ]

    data = None
    for path in paths:
        try:
            data = np.load(str(path / filename), allow_pickle=True)
            break
        except FileNotFoundError:
            continue

    if data is None:
        raise FileNotFoundError

    n_gal = len(data["labels"])
    np.random.seed(random_state)
    indices = np.random.permutation(n_gal)
    ind_split_train = int(np.ceil(frac_train * n_gal))
    ind_split_dev = ind_split_train + int(np.ceil(frac_dev * n_gal))

    # ind_bands = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4}
    # bands = params.get("bands", ("u", "g", "r", "i", "z"))
    # channels = np.array([ind_bands[band] for band, ind_band in zip(bands, ind_bands)])
    # slice_y, slice_x = crop_center(data["cube"].shape[1:3], params["image_shape"])
    # images = data["cube"][:, slice(*slice_y), slice(*slice_x), channels]
    # labels = consolidate_bins(data["labels"], n_bins_in=num_class, n_bins_out=num_class)
    images = data["cube"]
    labels = data["labels"]
    labels = keras.utils.to_categorical(labels, num_classes=num_class)
    z_spec = data["z"]
    cat = data["cat"]
    vals = pd.DataFrame()
    vals["u-g"] = (cat["modelMag_u"] - cat["extinction_u"]) - (
        cat["modelMag_g"] - cat["extinction_g"]
    )
    vals["g-r"] = (cat["modelMag_g"] - cat["extinction_g"]) - (
        cat["modelMag_r"] - cat["extinction_r"]
    )
    vals["r-i"] = (cat["modelMag_r"] - cat["extinction_r"]) - (
        cat["modelMag_i"] - cat["extinction_i"]
    )
    vals["i-z"] = (cat["modelMag_i"] - cat["extinction_i"]) - (
        cat["modelMag_z"] - cat["extinction_z"]
    )
    vals["EBV"] = cat["EBV"]
    vals["r"] = cat["cModelMag_r"] - cat["extinction_r"]

    scaler = StandardScaler()
    vals = scaler.fit_transform(np.array(vals))

    if params["logistic"]:
        z_spec = np.log((z_spec - params["z_min"]) / (params["z_max"] - z_spec))

    x_train = images[indices[:ind_split_train]] / float(image_scale)
    x_dev = images[indices[ind_split_train:ind_split_dev]] / float(image_scale)
    x_test = images[indices[ind_split_dev:]] / float(image_scale)

    y_train = labels[indices[:ind_split_train]]
    y_dev = labels[indices[ind_split_train:ind_split_dev]]
    y_test = labels[indices[ind_split_dev:]]

    z_spec_train = z_spec[indices[:ind_split_train]]
    z_spec_dev = z_spec[indices[ind_split_train:ind_split_dev]]
    z_spec_test = z_spec[indices[ind_split_dev:]]

    vals_train = vals[indices[:ind_split_train]]
    vals_dev = vals[indices[ind_split_train:ind_split_dev]]
    vals_test = vals[indices[ind_split_dev:]]

    if load_cat == False:
        return (
            (x_train, y_train, vals_train, z_spec_train),
            (x_dev, y_dev, vals_dev, z_spec_dev),
            (x_test, y_test, vals_test, z_spec_test),
        )
    if load_cat == True:
        cat_train = cat[indices[:ind_split_train]]
        cat_dev = cat[indices[ind_split_train:ind_split_dev]]
        cat_test = cat[indices[ind_split_dev:]]
        return (
            (x_train, y_train, vals_train, z_spec_train, cat_train),
            (x_dev, y_dev, vals_dev, z_spec_dev, cat_dev),
            (x_test, y_test, vals_test, z_spec_test, cat_test),
        )

def load_GOES(
    num_class,
    path_data,
    max_year_train=2014,
    max_year_dev=2017,
    random_state=200,
    image_scale=1.0,
    class_var='intensity',
    eye_check='include',
    **params
):
    filename_imgs = f"{params['dataset']}.npy"
    filename_aux = f"{params['dataset_aux']}.csv"
    
    paths = [
        Path(path_data),
        Path("/ocean/projects/dms190029p/gvincent/tc-rmw-data")
    ]

    data_img = None
    data_tc = None
    
    for path in paths:
        try:
            data_img = np.load(str(path / filename_imgs), allow_pickle=True)
            data_tc = pd.read_csv(str(path / filename_aux), parse_dates=['date'], dtype={'ID': 'str', 
                                                                                         'time_idx': 'int32',
                                                                                         'wind': 'float64',
                                                                                         'eye_diam': 'float64',
                                                                                         'rmw': 'float64',
                                                                                         'atcf': 'int8',
                                                                                         'lat': 'float64',
                                                                                         'lon': 'float64',
                                                                                         'pressure_min': 'float64',
                                                                                         'distance': 'float64',
                                                                                         'type': 'str',
                                                                                         'pressure_outer': 'float64',
                                                                                         'radius_outer': 'float64',
                                                                                         'eye': 'int8',
                                                                                         'synoptic': 'int8',
                                                                                         'eye_10': 'int8',
                                                                                         'eye_20': 'int8',
                                                                                         'eye_30': 'int8',
                                                                                         'category': 'str',
                                                                                         'category_num': 'int8',
                                                                                         'nan_frac': 'float64',})
            break
        except FileNotFoundError:
            continue

    if (data_img is None) or (data_tc is None):
        raise FileNotFoundError
    
    eye_check_vals = data_tc.eye.to_numpy()
    
    if eye_check == 'include':
        pass
    elif eye_check == 'exclude':
        indices = np.argwhere(np.logical_or(eye_check_vals == 0, eye_check_vals == 1)).flatten()
        data_tc = data_tc.iloc[indices].reset_index(drop = True)
        data_img = data_img[indices]
    elif eye_check == 'return':
        indices = np.argwhere(eye_check_vals == -99).flatten()
        data_tc = data_tc.iloc[indices].reset_index(drop = True)
        data_tc['eye'] = 0
        data_img = data_img[indices]
    else:
        raise ValueError("eye_check is not one of 'exclude', or 'return' ")
    
    years = data_tc.date.dt.year.to_numpy()
    
    train_idx = np.argwhere(years <= max_year_train).flatten()
    dev_idx = np.argwhere((years > max_year_train) & (years <= max_year_dev)).flatten()
    test_idx = np.argwhere(years > max_year_dev).flatten()
    
    if class_var == 'intensity':
        category_labels = data_tc["category_num"].to_numpy()
    elif class_var == 'eye':
        category_labels = data_tc["eye"].to_numpy()
    else:
        raise ValueError("class_var is not one of 'intensity' or 'eye' ")
    
    category_labels = keras.utils.to_categorical(category_labels, num_classes=num_class)
    
    rmw = data_tc["rmw"].to_numpy()
    aux_data = data_tc

    # Add extra dimension for image data to represent channels (remove if you actually have multiple channels):
    if len(data_img.shape) < 4:
        data_img = np.expand_dims(data_img, axis = 3)
    
    x_train = data_img[train_idx] / float(image_scale)
    x_dev = data_img[dev_idx] / float(image_scale)
    x_test = data_img[test_idx] / float(image_scale)

    y_train = category_labels[train_idx]
    y_dev = category_labels[dev_idx]
    y_test = category_labels[test_idx]

    rmw_train = rmw[train_idx]
    rmw_dev = rmw[dev_idx]
    rmw_test = rmw[test_idx]

    aux_train = aux_data.iloc[train_idx]
    aux_dev = aux_data.iloc[dev_idx]
    aux_test = aux_data.iloc[test_idx]
    
    return (
        (x_train, y_train, rmw_train, aux_train),
        (x_dev, y_dev, rmw_dev, aux_dev),
        (x_test, y_test, rmw_test, aux_test),
    )

def load_RMW(
    num_class,
    path_data,
    max_year_train=2014,
    max_year_dev=2017,
    random_state=200,
    image_scale=1.0,
    class_var = 'intensity',
    **params
):
    filename_imgs = f"{params['dataset']}.npy"
    filename_aux = f"{params['dataset_aux']}.csv"
    
    paths = [
        Path(path_data),
        Path("/ocean/projects/dms190029p/gvincent/tc-rmw-data")
    ]

    data_img = None
    data_tc = None
    
    for path in paths:
        try:
            data_img = np.load(str(path / filename_imgs), allow_pickle=True)
            data_tc = pd.read_csv(str(path / filename_aux), parse_dates=['date'], dtype={'ID': 'str', 
                                                                                         'time_idx': 'int32',
                                                                                         'wind': 'float64',
                                                                                         'eye_diam': 'float64',
                                                                                         'rmw': 'float64',
                                                                                         'atcf': 'int8',
                                                                                         'lat': 'float64',
                                                                                         'lon': 'float64',
                                                                                         'pressure_min': 'float64',
                                                                                         'distance': 'float64',
                                                                                         'type': 'str',
                                                                                         'pressure_outer': 'float64',
                                                                                         'radius_outer': 'float64',
                                                                                         'eye': 'int8',
                                                                                         'synoptic': 'int8',
                                                                                         'eye_10': 'int8',
                                                                                         'eye_20': 'int8',
                                                                                         'eye_30': 'int8',
                                                                                         'category': 'str',
                                                                                         'category_num': 'int8',
                                                                                         'nan_frac': 'float64',})
            break
        except FileNotFoundError:
            continue

    if (data_img is None) or (data_tc is None):
        raise FileNotFoundError
    
    rmw = data_tc["rmw"].to_numpy()
    rmw[np.isnan(rmw)] = -99
    data_tc.loc[np.isnan(data_tc.distance), 'distance'] = -99
    
    # Subset down to RMW that you want to train on
    rmw_inds = np.argwhere(
        (data_tc.atcf.to_numpy() == True) &
        (rmw > 6) &
        (rmw < 200) &
        (data_tc.wind.to_numpy() >= 35) &
        (data_tc.distance.to_numpy() >= 50) & 
        (data_tc.type.to_numpy() != 'E')
    ).flatten()
    
    data_tc = data_tc.iloc[rmw_inds].reset_index(drop = True)
    data_img = data_img[rmw_inds]
    
    rmw = data_tc["rmw"].to_numpy()
    
    years = data_tc.date.dt.year.to_numpy()
    
    train_idx = np.argwhere(years <= max_year_train).flatten()
    dev_idx = np.argwhere((years > max_year_train) & (years <= max_year_dev)).flatten()
    test_idx = np.argwhere(years > max_year_dev).flatten()
    
    if class_var == 'intensity':
        category_labels = data_tc["category_num"].to_numpy()
    elif class_var == 'eye':
        category_labels = data_tc["eye"].to_numpy()
    else:
        raise ValueError("class_var is not one of 'intensity' or 'eye' ")
    
    category_labels = keras.utils.to_categorical(category_labels, num_classes=num_class)

    aux_data = data_tc
    
    # Get extra data for regression
    ebt_regression_data = aux_data[['lat', 'lon', 'wind', 'pressure_min', 'distance']].copy()
    
    # Do some transfomations so these variables are approximately normal and unit variance
    ebt_regression_data.wind = np.log(ebt_regression_data.wind)
    ebt_regression_data.pressure_min = -1 * np.log(np.max(ebt_regression_data.pressure_min) + 5 - ebt_regression_data.pressure_min)
    ebt_regression_data.distance = np.log(ebt_regression_data.distance)
    ebt_regression_data = ebt_regression_data.to_numpy()
    scaler = StandardScaler()
    ebt_regression_data = scaler.fit_transform(ebt_regression_data)
    
    # Add extra dimension for image data to represent channels (remove if you actually have multiple channels):
    if len(data_img.shape) < 4:
        data_img = np.expand_dims(data_img, axis = 3)
    
    x_train = data_img[train_idx] / float(image_scale)
    x_dev = data_img[dev_idx] / float(image_scale)
    x_test = data_img[test_idx] / float(image_scale)

    y_train = category_labels[train_idx]
    y_dev = category_labels[dev_idx]
    y_test = category_labels[test_idx]

    rmw_train = rmw[train_idx]
    rmw_dev = rmw[dev_idx]
    rmw_test = rmw[test_idx]

    aux_train = aux_data.iloc[train_idx]
    aux_dev = aux_data.iloc[dev_idx]
    aux_test = aux_data.iloc[test_idx]
    
    ebt_train = ebt_regression_data[train_idx]
    ebt_dev = ebt_regression_data[dev_idx]
    ebt_test = ebt_regression_data[test_idx]
    
    return (
        (x_train, y_train, rmw_train, ebt_train, aux_train),
        (x_dev, y_dev, rmw_dev, ebt_dev, aux_dev),
        (x_test, y_test, rmw_test, ebt_test, aux_test),
    )



class DataGenerator(keras.utils.Sequence):
    """Generates custom batches.
    
    Designed to implement oversampling of underrepresented classes.
    
    class_weights should be a vector of length ncapsule. A value of 1.0 corresponds to 
    the number of samples in the majority class, so class_weights = [1.0, 1.0, 1.0] means
    that each class is represented in the upsampled dataset with the same number of 
    datapoints that the majority class has. Another example: [2.0, 1.0, 1.0] means
    that class 0 will have 2x as many datapoints in the sample as either class 1 or 
    class 2, and class 0 will have 2x as many datapoints as the original majority class.
    
    Classes in y_train should be integers starting at zero and increasing by 1 for each class.
    The order of these classes corresponds to the order of the weights given in `class_weights`.
    """

    def __init__(
        self,
        x_train,
        y_train,
        batch_size,
        class_weights = [1.0, 1.0, 1.0],
        capsnet = True
    ):
        "Initialization"
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.weights = class_weights
        self.capsnet = capsnet
        
        labels = np.argmax(y_train, axis = 1)
        self.classes = np.sort(np.unique(labels))
        self.class_counts = [np.sum(labels == c) for c in self.classes]
        self.n_max = np.max(self.class_counts)
        self.class_max = np.argmax(self.class_counts)
        self.class_indices = [np.argwhere(labels == c).flatten() for c in self.classes]
        
        assert len(self.classes) == len(self.weights), f"Number of classes in y_train ({len(self.classes)}) does not match number of provided weights ({len(self.weights)})."
        
        self.epoch = 0
        self.on_epoch_end()

    def __len__(self):
        "calculates the number of batches per epoch"
        return int(np.ceil(len(self.balanced_indices) / self.batch_size))

    def __getitem__(self, idx):
        "Generate one batch of data"
        # Generate indexes of the batch
        batch_indices = self.balanced_indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]  # uniform batching

        # Generate data
        batch_x = self.x_train[batch_indices]
        batch_y = self.y_train[batch_indices]
        
        if self.capsnet:
            return [batch_x, batch_y], [batch_y, batch_x]
        else:
            return batch_x, batch_y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.epoch += 1
        
        balanced_class_indices = []
        for c, cidx in zip(self.classes, self.class_indices):
            cidx_upsampled = np.resize(cidx, int(self.n_max * self.weights[c]))
            balanced_class_indices.append(cidx_upsampled)
        
        self.balanced_indices = np.random.permutation(np.concatenate(balanced_class_indices))