void preProcess(uchar4 **inputImage, unsigned char **greyImage,
				uchar4 **d_rgbaImage, unsigned char **d_greyImage,
				const std::string &filename) {
	//ensure the file initializes fine
	checkCudaErrors(cudaFree(0));

	cv.
}