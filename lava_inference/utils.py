import pandas as pd
import numpy as np


def generate_movement_name_df():
    """
    Generate a DataFrame containing movement names and corresponding session information.

    Returns:
    df (pandas.DataFrame): DataFrame with columns ['session', 'mov', 'mov_string'].

    Example usage:
    >>> df = generate_movement_name_df()
    >>> print(df.head())

       session  mov            mov_string
    0        1    1    Left arm abduction
    1        1    2   Right arm abduction
    2        1    3    Left leg abduction
    3        1    4   Right leg abduction
    4        1    5   Left arm bicep curl

    """
    df = pd.DataFrame(columns=['session', 'mov', 'mov_string'])
    data_s1 = {'session': [1, 1, 1, 1, 1, 1, 1, 1], \
        'mov': [1, 2, 3, 4, 5, 6, 7, 8], \
        'mov_string': ['Left arm abduction', 'Right arm abduction', 'Left leg abduction', 'Right leg abduction', 'Left arm bicep curl', 'Right arm bicep curl', 'Left leg knee lift', 'Right leg knee lift']}
    
    data_s2 = {'session': [2, 2, 2, 2, 2, 2], \
        'mov': [1, 2, 3, 4, 5, 6], \
        'mov_string': ['Walking 3.5 km/h', 'Single jump up-down', 'Single jump forwards', 'Multiple jumps up-down', 'Hop right foot', 'Hop left foot']}
    
    data_s3 = {'session': [3, 3, 3, 3, 3, 3], \
        'mov': [1, 2, 3, 4, 5, 6], \
        'mov_string': ['Punch straight forward left', 'Punch straight forward right', 'Punch up forwards left', 'Punch up forwards right', 'Punch down forwards left', 'Punch down forwards right']}
    
    data_s4 = {'session': [4, 4, 4, 4, 4, 4], \
        'mov': [1, 2, 3, 4, 5, 6], \
        'mov_string': ['Slow jogging 7 km/h', 'Star jumps', 'Kick forwards left', 'Kick forwards right', 'Side kick forwards left', 'Side kick forwards right']}
    
    data_s5 = {'session': [5, 5, 5, 5, 5, 5, 5], \
        'mov': [1, 2, 3, 4, 5, 6, 7], \
        'mov_string': ['Wave hello left hand', 'Wave hello right hand', 'Circle left hand', 'Circle right hand', 'Figure-8 left hand', 'Figure-8 right hand', 'Clap']}
    

    df = pd.DataFrame(data=data_s1)
    df = pd.concat([df, pd.DataFrame(data=data_s2)], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(data=data_s3)], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(data=data_s4)], ignore_index=True)
    df = pd.concat([df, pd.DataFrame(data=data_s5)], ignore_index=True)
    
    return df


def get_xy_coords_for_prediction_and_target(prediction, target, joint, downsample_factor=2, img_height=260, img_width=344):
    """
    Calculates the predicted and target coordinates for a specific joint in a hand detection model.

    Args:
        prediction (numpy.ndarray): The prediction output from the model.
        target (numpy.ndarray): The target data.
        joint (int): The index of the joint.
        downsample_factor (int, optional): The factor by which the image is downsampled. Defaults to 2.
        img_height (int, optional): The height of the image. Defaults to 260.
        img_width (int, optional): The width of the image. Defaults to 344.

    Returns:
        tuple: A tuple containing the predicted y-coordinate, predicted x-coordinate, target y-coordinate, and target x-coordinate.
    """


    xy_coord_vec_length = int((img_width + img_height)/downsample_factor) # 302

    coords_1hot = prediction[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
    coords_one_hot_y1 = coords_1hot[0:int(img_height / downsample_factor)]
    coords_one_hot_x1 = coords_1hot[int(img_height / downsample_factor):]
    
    # get prediction from model output
    predicted_y1 = np.where(coords_one_hot_y1 == coords_one_hot_y1.max())[0][0]
    predicted_x1 = np.where(coords_one_hot_x1 == coords_one_hot_x1.max())[0][0]
    predicted_y1 = predicted_y1 * downsample_factor
    predicted_x1 = predicted_x1 * downsample_factor

    # get parts of the target data by coordinate and joint
    target_1hot = target[joint*xy_coord_vec_length:(joint+1)*xy_coord_vec_length]
    target_one_hot_y1 = target_1hot[0:int(img_height / downsample_factor)]
    target_one_hot_x1 = target_1hot[int(img_height / downsample_factor):]
    
    # get prediction from target
    target_y1 = np.where(target_one_hot_y1 == target_one_hot_y1.max())[0][0]
    target_x1 = np.where(target_one_hot_x1 == target_one_hot_x1.max())[0][0]    
    target_y1 = target_y1 * downsample_factor
    target_x1 = target_x1 * downsample_factor

    return predicted_y1, predicted_x1, target_y1, target_x1


def comp_dist_for_sample(prediction, target, downsample_factor=2, img_height=260, img_width=344):
    """
    Computes the Euclidean distance between the predicted and target coordinates for each joint in a hand detection model.

    Args:
        prediction (numpy.ndarray): The prediction output from the model.
        target (numpy.ndarray): The target data.
        downsample_factor (int, optional): The factor by which the image is downsampled. Defaults to 2.
        img_height (int, optional): The height of the image. Defaults to 260.
        img_width (int, optional): The width of the image. Defaults to 344.

    Returns:
        tuple: A tuple containing the distance between the predicted and target coordinates for each joint.

    Example usage:
    >>> prediction = np.array(...)
    >>> target = np.array(...)
    >>> dist1, dist2 = comp_dist_for_sample(prediction, target)
    >>> print(dist1, dist2)
        """
    
    # prediction.shape # (604,0)
    # target.shape # (604,0)
    # xy_coord_vec_length = int((img_width + img_height)/downsample_factor) # 302

    # get xy coords of prediction and target per joint
    predicted_y1, predicted_x1, target_y1, target_x1 = get_xy_coords_for_prediction_and_target(prediction, target, joint=0, downsample_factor=2, img_height=260, img_width=344)
    predicted_y2, predicted_x2, target_y2, target_x2 = get_xy_coords_for_prediction_and_target(prediction, target, joint=1, downsample_factor=2, img_height=260, img_width=344)

    # compute distance between prediction and target
    dist1 = np.sqrt((predicted_y1 - target_y1)**2 + (predicted_x1 - target_x1)**2)
    dist2 = np.sqrt((predicted_y2 - target_y2)**2 + (predicted_x2 - target_x2)**2)

    return dist1, dist2
        
