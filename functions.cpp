void preProcess(uchar4 **inputImage, unsigned char **greyImage,
				uchar4 **d_rgbaImage, unsigned char **d_greyImage,
				const std::string &filename) {
	//ensure the file initializes fine
	checkCudaErrors(cudaFree(0));

	cv::Mat image;
	image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty()) {
		std::cerr << "Couldn't open file: " << filename << std::endl;
		exit(1);
	}

	cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

	//allocate memory for output
	imageGrey.create(image.rows, image.cols, CV_8UC1);

	//This shouldn't ever happen given the way the images are created
  	//at least based upon my limited understanding of OpenCV, but better to check
  	if (!imageRGBA.isContinuous() || !imageGrey.isContinuous()) {
    	std::cerr << "Images aren't continuous!! Exiting." << std::endl;
    	exit(1);
  	}

  	*inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
  	*greyImage  = imageGrey.ptr<unsigned char>(0);

	const size_t numPixels = numRows() * numCols();
  	//allocate memory on the device for both input and output
  	checkCudaErrors(cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels));
  	checkCudaErrors(cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels));
  	checkCudaErrors(cudaMemset(*d_greyImage, 0, numPixels * sizeof(unsigned char))); //make sure no memory is left laying around

  	//copy input array to the GPU
  	checkCudaErrors(cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

  	d_rgbaImage__ = *d_rgbaImage;
  	d_greyImage__ = *d_greyImage;
}