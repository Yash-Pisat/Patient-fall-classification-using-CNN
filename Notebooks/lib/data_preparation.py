import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt

from lib import Augmentation_methods

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from IPython.display import clear_output

from scipy.signal import stft

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


from pathlib import Path
import datetime
import glob as gl

def CalculateFeatures(A: np.array, return_numpy=False):
    """
    This function calculates various features of a given numpy array 'A' such as the maximum value, median value, mean value, and 25th and         75th quantiles. The elements in the array 'A' are first squared and then the features are calculated based on the transformed array.

    Parameters:
    A (np.array): The input numpy array.
    return_numpy (bool): Specifies whether the result should be returned as a numpy array or a dictionary. Defaults to False.

    Returns:
    np.array or dict: The calculated features returned as a numpy array or a dictionary, based on the value of 'return_numpy'. If return_numpy     is set to True, the result is returned as a numpy array with elements as [maximum, median, mean, quantile25, quantile75]. If return_numpy     is set to False, the result is returned as a dictionary with keys 'maximum', 'median', 'mean', 'quantile25', and 'quantile75'.
    
    """
    # Square all elements in array to calculate the features
    A = np.square(A - np.mean(A))
    
    max_val = np.max(A)
    median_val = np.median(A)
    mean_val = np.mean(A)
    q_25 = np.quantile(A, 0.25)
    q_75 = np.quantile(A, 0.75)

    if return_numpy:
        return np.array([max_val, median_val, mean_val, q_25, q_75])
    return {'maximum': max_val, 'median': median_val, 'mean': mean_val, 'quantile25': q_25, 'quantile75': q_75}

def RandomInterval(df: pd.DataFrame, interval_time: int = 10, sampling_rate: int = 1600,
                     return_values: np.array = ['z'], seed: int or None = None) -> np.array:
    """
    This function returns a random interval of a given dataframe with a specified interval time, sampling rate, and selected values.

    :param df: pd.DataFrame - input dataframe
    :param interval_time: int - the length of the interval in seconds (default = 10)
    :param sampling_rate: int - the number of samples per second (default = 1600)
    :param return_values: np.array - the columns of the dataframe to return as a numpy array (default = ['z'])
    :param seed: int or None - seed value for the random number generator (default = None)
    :return: np.array - the selected interval of the dataframe as a numpy array
    
    """
    if seed:
        np.random.seed(seed)
    # approximate number of points that correspond to interval_time
    number_of_points = interval_time / (1 / sampling_rate)
    # last possible beginning of the time interval in order for it to have its full length
    last_possible_start = len(df) - number_of_points
    # Assumption: it does not matter if there are partially overlapping samples -> Randomly draw with no restrictions
    starting_point = np.random.randint(low=0, high=last_possible_start)
    end_point = int(starting_point + number_of_points)

    # only return the requested values of the dataframe as np.array
    return np.array(df[starting_point:end_point][return_values]).reshape(-1)

def _read_binary_to_pandas(filepath, dt=np.dtype('int16'), datetime_fmt=True):
    
    """
    This function reads a binary file and converts its contents into a pandas dataframe. The contents of the file are divided into three           arrays x, y, and z. The data is then reordered and time is added based on the filename. The datetime format can be specified by the           argument "datetime_fmt". The final output is a pandas dataframe containing the columns "z" and "time".

    Parameters:
    filepath (str): The path to the binary file to be read.
    dt (np.dtype, optional): The data type of the binary file contents. Defaults to np.dtype('int16').
    datetime_fmt (bool, optional): Specifies if the "time" column should be in datetime format or not. Defaults to True.

    Returns:
    df (pandas dataframe): A pandas dataframe containing the columns "z" and "time".
    
    """
    try:
        df = pd.DataFrame(np.fromfile(filepath, dtype=dt, count=-1, sep=''))

    except IOError:
        print("Error while opening the file!")

    # We need to reorder data & and add time from filename
    x = df.iloc[0::3].to_numpy().reshape(-1)
    y = df.iloc[1::3].to_numpy().reshape(-1)
    z = df.iloc[2::3].to_numpy().reshape(-1)

    min_len = np.min([len(x), len(y), len(z)])
    samples_per_second = 1600

    print(f"{min_len} rows from filepath: {filepath.stem}")

    df = pd.DataFrame({'x': x[:min_len], 'y': y[:min_len], 'z': z[:min_len]})
    df.name = filepath.stem

    if datetime_fmt:
        start = datetime.datetime.strptime(filepath.stem, '%Y-%m-%d_%H-%M-%S')
        end = start + datetime.timedelta(seconds=(1 / samples_per_second) * (min_len - 1))
        df['time'] = pd.date_range(start, end, freq="0.625ms")
    else:
        start = 0
        end = (1 / samples_per_second) * (min_len - 1)
        df['time'] = np.linspace(0, end, min_len)

    return df[["z", "time"]]    

def load_real_data(room: str ="101 B1", 
                   year: int =2022,
                   month: int =9,
                   day: int =10,
                   startHour: int =12,
                   lastHour: int =13,
                   logging_level="WARNING") -> pd.DataFrame:
    
    """
    This function load_real_data loads real data for a given room, date, and time range. It returns the data as a pandas DataFrame. The           function reads binary files from a specific directory, sorts them, filters them based on the given date and time range, and concatenates       the data into a single DataFrame.
    
    """
    
    # Global directory path + the specific room number
    DATA_PATH = pathlib.Path("/store/projects/fallki/MessdatenFeldversuch/gesamt", room)

    # Collecting all file pathes and sort them with ascending time
    runFiles = []
    runFiles += DATA_PATH.glob(f'*.bin')
    runFiles.sort()

    # Creating a Dataframe with one column full path and one column only the timestamp
    file_path = [f.stem for f in runFiles]
    design = {"path": runFiles, "date": file_path}
    df_file_path = pd.DataFrame(design)

    # Converting the "date" column to a datetime format
    df_file_path["date"] = pd.to_datetime(df_file_path['date'], format='%Y-%m-%d_%H-%M-%S')
    # Interpreting the arguments as start and end time
    start = pd.Timestamp(year=year, month=month, day=day, hour=startHour)
    end = pd.Timestamp(year=year, month=month, day=day, hour=lastHour)

    # Creating a mask for ovelaying the df_file_path
    mask = (df_file_path['date'] >= start) & (df_file_path['date'] <= end)
    df_file_path = df_file_path.loc[mask]

    # Loading the data in the specified time range
    df = pd.DataFrame()
    for index, row in df_file_path.iterrows():
        temp = _read_binary_to_pandas(row["path"])
        df = pd.concat([df, temp])

    return df.reset_index(drop=True)

def fake_lab_data():
    """
    This function prepares the features and labels for positive, negative and the dummy data.
    
    """
    BASEDIR = pathlib.Path('/store/projects/fallki/')
    LONGTIMEDIR = pathlib.Path('Langzeitmessung/2/')
    # one sensor was applied to the floor and was applied to the bed
    df = pd.read_parquet(BASEDIR / LONGTIMEDIR / 'standalone.parquet') 
    df1 = pd.read_parquet(BASEDIR / LONGTIMEDIR /'bett.parquet') 

    Negative_Events_Data = []
    Negative_Events = []
    Negative_Beschreibungen = 20000 * ["Leer"]

    # get 20000 samples and negative training samples and calculate there features.
    for i in range(10000):
        Interval_Bett = RandomInterval(df)
        Interval_Boden = RandomInterval(df1)
        Negative_Events_Data.append(Interval_Bett)
        Negative_Events_Data.append(Interval_Boden)
        Features_Bett = CalculateFeatures(Interval_Bett)
        Features_Boden = CalculateFeatures(Interval_Boden)
        Negative_Events.append(np.array(list(Features_Bett.values())))
        Negative_Events.append(np.array(list(Features_Boden.values())))
    
    Negative_Labels = len(Negative_Events) * [[0]]

    # -------Positive Events------------   
    # load data
    FAKEDIR = pathlib.Path('Fake-Events/2. Fake-Events Messung/')

    df_fake = pd.read_parquet(BASEDIR / FAKEDIR / 'gSensoren_FakeEvents.parquet')
    df_fake_groups = df_fake.groupby(['teensy', 'experiment', 'run', 'rep'])
    event_measurements = df_fake[['teensy', 'experiment', 'run', 'rep']].drop_duplicates()

    Positive_Events_Data = []
    Positive_Events = []
    Positive_Beschreibungen = []

    for _, measurement in event_measurements.iterrows():
        data = df_fake_groups.get_group((measurement['teensy'], measurement['experiment'], measurement['run'], measurement['rep']))
        Positive_Events_Data.append(np.array(data['z']))
        Positive_Beschreibungen.append(np.array(data['experiment'][0]))
        Positive_Events.append(CalculateFeatures(np.array(data['z'])))
    
    # Labels for first step:
    Positive_Labels = np.array(len(Positive_Events) * [[1]])
    Positive_Events = pd.DataFrame.from_dict(Positive_Events).to_numpy()

    Dummy = np.where(event_measurements['experiment'] == 'Dummy')[0]

    # Labels for second step (not relevant for this notbook):
    Dummy_Labels = np.zeros(len(event_measurements))
    Dummy_Labels[Dummy] = 1 

    # ------Combine Positve, Negative events to one array---------
    All_Events = np.concatenate((Negative_Events, Positive_Events))
    All_Beschreibungen = Negative_Beschreibungen + Positive_Beschreibungen 

    All_Labels = np.concatenate((Negative_Labels, Positive_Labels))
    All_Events_Data = Negative_Events_Data + Positive_Events_Data 
    All_Dummy_Labels = np.concatenate((Negative_Labels, Dummy_Labels.reshape(1079,1)))

    shuffler = np.random.permutation(len(All_Events))
    All_Events = All_Events[shuffler]
    All_Labels = All_Labels[shuffler]

    #Shuffle events:
    shuffler = np.random.permutation(len(All_Events))
    All_Events = np.take(All_Events, shuffler, axis=0)
    All_Labels = All_Labels[shuffler]
    All_Beschreibungen = np.array(All_Beschreibungen)[shuffler]
    All_Dummmy_Labels = All_Dummy_Labels[shuffler]
    All_Events_Data = np.array(All_Events_Data,dtype=object)[shuffler]

    # Derive Labels for Fake Events
    All_Fake_Labels = np.zeros(All_Dummmy_Labels.shape)
    All_Fake_Labels[(All_Dummmy_Labels == 0).reshape(-1) & (All_Labels == 1).reshape(-1)] = 1

    #---------------Dummy Data-------------------------------
    df_fake_dummy = df_fake[df_fake.experiment == 'Dummy']
    df_fake_dummy_group = df_fake_dummy.groupby(['teensy', 'experiment', 'run', 'rep'])

    event_measurements = df_fake_dummy[['teensy', 'experiment', 'run', 'rep']].drop_duplicates()

    df_fake_dummy_Data = []
    df_fake_dummy_Events = []
    df_fake_dummy_Beschreibungen = []

    for _, measurement in event_measurements.iterrows():
        data = df_fake_dummy_group.get_group((measurement['teensy'], measurement['experiment'], measurement['run'], measurement['rep']))
        df_fake_dummy_Data.append(np.array(data['z']))
        df_fake_dummy_Beschreibungen.append(np.array(data['experiment'][0]))
        df_fake_dummy_Events.append(CalculateFeatures(np.array(data['z'])))

    # Labels for first step:
    df_fake_dummy_Labels = np.array(len(df_fake_dummy_Events) * [[1]])
    df_fake_dummy_Data = np.array(df_fake_dummy_Data,dtype=object)

    return df_fake_dummy_Data, All_Events, All_Labels,Positive_Events_Data,Positive_Beschreibungen,df_fake

def shifting_window_segmentation(df: pd.DataFrame, 
                                 interval_time: int = 10, 
                                 sampling_rate: int = 1600,
                                 stepsize: int = 1, 
                                 labeled_data: bool = False,
                                 time_column_name: str = 'time', 
                                 data_column_name: str = 'z',
                                 label_column_name: str = 'label',
                                 binary_column_name: str = 'binary_label',) -> pd.DataFrame:
    """
    shifting_window_segmentation is a function that takes in a pandas DataFrame (df) and splits the data into overlapping chunks.

    Parameters:
    df (pd.DataFrame) - A DataFrame containing the time, data, and optionally the label and binary label of the data
    interval_time (int) - The time interval (in seconds) to be used to split the data (default is 10)
    sampling_rate (int) - The number of samples per second in the data (default is 1600)
    stepsize (int) - The step size (in seconds) to be used when splitting the data (default is 1)
    labeled_data (bool) - A flag indicating if the data is labeled or not (default is False)
    time_column_name (str) - The name of the column in the DataFrame containing the time data (default is 'time')
    data_column_name (str) - The name of the column in the DataFrame containing the data (default is 'z')
    label_column_name (str) - The name of the column in the DataFrame containing the label (default is 'label')
    binary_column_name (str) - The name of the column in the DataFrame containing the binary label (default is 'binary_label')

    Returns:
    returns a pandas DataFrame with columns for the left and right time of each chunk, the chunk of data, the features calculated from the         chunk, and optionally the label and binary label of the chunk.

    """

    if interval_time == stepsize:
        print("Attention, interval time equals step size causesv"
              "missing overlap of data: Chunk Mode activated!")

    left_pointer, right_pointer = 0, (interval_time * sampling_rate)

    results = {"left_t": [], "right_t": [], "z": [], "features": []}

    if labeled_data:
        results['label'] = []
        results['binary_label'] = []

    while right_pointer < len(df['z']):

        results["left_t"].append(df[time_column_name].iloc[left_pointer])
        results["right_t"].append(df[time_column_name].iloc[right_pointer])
        results["z"].append(np.array(df[data_column_name].iloc[left_pointer:right_pointer]))

        tmp_feature = CalculateFeatures(np.array(
            df[data_column_name].iloc[left_pointer:right_pointer]), return_numpy=True)
        tmp_feature = np.array([tmp_feature])
        results["features"].append(tmp_feature)

        if labeled_data:
            results['label'].append(df[label_column_name].iloc[left_pointer])
            results['binary_label'].append(df[binary_column_name].iloc[left_pointer])

        shift = sampling_rate * stepsize

        left_pointer += shift
        right_pointer += shift

    return pd.DataFrame(results)    


def features_and_target(df_list):
    """
    This function takes in a list of dataframes and returns a list of tuples. Each tuple contains the features and target data. The features       data is obtained by stacking the "normalizedSpectrum" column of the dataframe. The target data is obtained by converting the     istSturz     column to a numpy array.

    Parameters:
    df_list - list of dataframes, or a single dataframe

    Returns:
    x_y_sets - list of tuples, each containing the features and target data in the form (X, y)
    
    """
    x_y_sets = []
    if isinstance(df_list, list):
        for df in df_list:
            X = np.stack(df["normalizedSpectrum"].to_list())
            y = df["istSturz"].to_numpy()
            x_y_sets.append((X, y))
    else:
        X = np.stack(df_list["normalizedSpectrum"].to_list())
        y = df_list["istSturz"].to_numpy()
        x_y_sets.append((X, y))
    return x_y_sets


def SignalToSpectrum(signal, samplerate=1600):
    """
    This function converts a given signal into its spectrum representation.

    Parameters:
    signal (array-like): The input signal to be converted into spectrum.
    samplerate (int, optional): The sample rate of the input signal. Default value is 1600.

    Returns:
    A numpy array containing the spectrum representation of the input signal with the shape (F x T x 1),
    where F is the number of frequency bins and T is the number of time frames.

    """
    SIGNAL_LENGTH = samplerate * 10
    F = stft(signal[:SIGNAL_LENGTH], 1600, nperseg=128)[2]
    F = F[2:,]
    return np.expand_dims(np.abs(F), axis=2) # Add extra dimension for TF Conv2D



def dataframe_prep(df):
    """
    The function converts the list of dictionaries into a new pandas dataframe with the specified columns.
    
    Parameters:
    df (pandas dataframe): A dataframe containing data to be processed.

    Returns:
    result_dataframe (pandas dataframe): A new dataframe containing event type, istSturz, signal length, and spectrum information.
    normalization_factor (float): The maximum value of the original spectrum used for normalization.
    
    """
    M = []
    for i in range(len(df)):
        M.append({"event": "Dummy" ,
                  "istSturz": 1,
                  "signalLenght": len(df.z[i]),
                  "spectrum": SignalToSpectrum(df.z[i])})
        result_dataframe = pd.DataFrame.from_dict(M)
        result_dataframe["normalizedSpectrum"] = result_dataframe["spectrum"] / result_dataframe["spectrum"].map(np.max).max()
        normalization_factor = result_dataframe["spectrum"].map(np.max).max() 
        
    return result_dataframe, normalization_factor


def vae_prep(augmented_data, normalization_factor):
    """
    Prepare the data for Variational Autoencoder (VAE) model.

    This function takes two arguments:
    augmented_data: A list of numpy arrays representing the augmented signal data.
    normalization_factor: A scalar representing the normalization factor to be applied to the augmented_data.

    It returns a pandas DataFrame 'vae_df' with the following columns:
    event: A string column with a fixed value of "Dummy".
    istSturz: A binary column indicating if the event is a fall (1) or not (0).
    signalLength: An integer column representing the length of the signal.
    spectrum: A numpy array column representing the augmented signal data multiplied by the normalization factor.
    normalizedSpectrum: A numpy array column representing the augmented signal data.

    """
    
    V = []
    for i in range(len(augmented_data)):
        V.append({"event": "Dummy" ,
                  "istSturz": 1,
                  "signalLenght": 16000,
                  "spectrum": normalization_factor * augmented_data[i],
                  "normalizedSpectrum":augmented_data[i]})
    vae_df = pd.DataFrame.from_dict(V)
    return vae_df


def build_model():
    """
    This function build_model creates and returns a Sequential model in Keras with a Conv2D layer, MaxPooling2D layer, Flatten layer, and         Dense layer.The model is compiled with the Adam optimizer, binary crossentropy loss, and accuracy metric.
    
    """
    mdl = Sequential()
    mdl.add(Conv2D(8, (63, 5), activation="relu", input_shape=(63, 251, 1)))
    mdl.add(MaxPooling2D((1, 4)))
    mdl.add(Flatten())
    mdl.add(Dense(1))
    mdl.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return mdl