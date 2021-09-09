from inference import inference_image
    

if __name__ == '__main__':
    image_path = './dataset/test_img/000066_0.jpg'
    clothes_path = './dataset/test_clothes/017575_1.jpg'
    edge_path = './dataset/test_edge/017575_1.jpg'
    input_path = { 'image': image_path,'clothes': clothes_path ,'edge': edge_path}
    result_path = inference_image(input_path , "./result.jpg")