BASE_PATH = '/home/diam/Desktop/WBC_Test_Images/'

def get_filename_for_index(index):
    PREFIX = 'Original_Images/BloodImage_'
    num_zeros = 5 - len(index)
    path = '0' * num_zeros + index
    return PREFIX + path + '.jpg'


reader = csv.reader(open(BASE_PATH + 'labels.csv'))
# skip the header
next(reader)

X = []
y = []

for row in reader:
    label = row[2]
    if len(label) > 0 and label.find(',') == -1 and label is not 'BASOPHIL':
        filename = get_filename_for_index(row[1])
        img_file = cv2.imread(BASE_PATH + filename)
        if img_file is not None:
            img_file = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
            img_file = scipy.misc.imresize(arr=img_file, size=(120, 160, 3))
            img_arr = np.asarray(img_file)
            X.append(img_arr)
            y.append(label)
        else:
            print("No file found", BASE_PATH + filename)


X = np.asarray(X)
y = np.asarray(y)

#print(X,y)

##train_test_split from scikit learn. t_t_s(*arrays, **options)
#test_size = if float, should be b/w 0.0-1.0, represents proportion dataset
#included in train size. default = 0.25
#random_state = int (seed). default = NONE.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


##make 4 lists of subset x values for each bc type in the x_train dataset 
eosinophil_samples = X_train[np.where(y_train == 'EOSINOPHIL')]
lymphocyte_samples = X_train[np.where(y_train == 'LYMPHOCYTE')]
monocyte_samples = X_train[np.where(y_train == 'MONOCYTE')]
neutrophil_samples = X_train[np.where(y_train == 'NEUTROPHIL')]


#old version
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



#def get_data(folder):
#    """
#    Load the data and labels from the given folder.
#    """
#    X = []
#    y = []
#    filenames = []

#    for wbc_type in os.listdir(folder):
#        if not wbc_type.startswith('.'):
#            if wbc_type in ['NEUTROPHIL', 'EOSINOPHIL']:
#                label = 'POLYNUCLEAR'
#            else:
#                label = 'MONONUCLEAR'
#            for image_filename in os.listdir(folder + wbc_type):
#		filename = folder + wbc_type + '/' + image_filename
#                img_file = cv2.imread(folder + wbc_type + '/' + image_filename)
#                if img_file is not None:
#                    # Downsample the image to 120, 160, 3
#                    img_file = scipy.misc.imresize(arr=img_file, size=(120, 160, 3))
#                    img_arr = np.asarray(img_file)
#                    X.append(img_arr)
#                    y.append(label)
#		    filenames.append(filename)
	
#    X = np.asarray(X)
#    y = np.asarray(y)
#    return X,y,filenames

