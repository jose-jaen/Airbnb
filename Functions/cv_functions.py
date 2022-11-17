# Required packages
from deepface import DeepFace
import gender_guesser.detector as gender
from tensorflow.keras.utils import get_file

def cv_model(data):
    """ Performs transfer learning to predict gender from images with a CV Model

    - Parameters:
        - data = Dataset where links to images are stored
    
    - Output:
        - CV Model gender prediction dataset
    """
    # Conditions for multiple hosts (couples/firms) or hosts with no picture
    cond1 = data['host_has_profile_pic'] == 1
    cond2 = ~data['host_name'].str.contains('&')
    cond3 = ~data['host_name'].str.contains('Or ', case=True)
    cond4 = ~data['host_name'].str.contains('And ', case=True)
    cond5 = ~data['host_name'].str.contains('AND', case=True)
    cond6 = ~data['host_name'].str.contains('Randy\\\Zuke')
    cond7 = ~data['host_name'].str.contains('\+', case=True)

    # Filter dataset
    cv_data = data[['id', 'host_picture_url']][cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7]

    # Create new gender column for predictions
    cv_data['cv_gender'] = cv_data['id']

    # Vector of internet links to pictures
    links = [i for i in cv_data['host_picture_url']]

    # Make gender predictions with CV Model
    for i in range(len(cv_data['id'])):
        try:
            # Hosts that still have a profile picture are passed through a CV Model
            host_pic = get_file('host' + str(i) + '.jpg', links[i])
            result = DeepFace.analyze(host_pic, actions=['gender'], enforce_detection=False)
            cv_data['cv_gender'][i] = result['gender']
        except:
            # Hosts with no profile picture are passed through an NLP Model
            cv_data['cv_gender'][i] = 0
            continue
    return cv_data
