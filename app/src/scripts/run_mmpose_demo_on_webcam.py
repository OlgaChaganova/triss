from mmpose.apis import MMPoseInferencer

img_path = '/home/olga/PycharmProjects/yolov8/test3.jpg'   # replace this with your own image path

# instantiate the inferencer using the model alias
inferencer = MMPoseInferencer('animal')

# The MMPoseInferencer API employs a lazy inference approach,
# creating a prediction generator when given input
result_generator = inferencer(img_path, show=True)
result = next(result_generator)
print(result)
