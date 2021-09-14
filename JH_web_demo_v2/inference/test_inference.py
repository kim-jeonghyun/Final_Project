from inference import inference_image
    

if __name__ == '__main__':
    image_path = './dataset/test_img/female_model.jpg'
    clothes_path = './dataset/test_clothes/0988551_02.jpeg'
    edge_path = './dataset/test_edge/0988551_02.jpeg'
    input_path = { 'image': image_path,'clothes': clothes_path ,'edge': edge_path}
    result_path = inference_image(input_path , "./result.jpg")