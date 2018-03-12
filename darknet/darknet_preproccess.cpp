#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>

using namespace cv;

// convert
//     1. change from CV_8UC3 to float image(CV_32FC3)
//     2. normalize with /255.
//     3. change the order from HWC to CHW
void convert_to_float(Mat src, Mat dst)
{
    unsigned char *src_data = (unsigned char *)src.data;
    int h = src.rows;
    int w = src.cols;
    int c = src.channels();
    int step = src.step1();
    int i, j, k;

    for(i = 0; i < h; ++i)
    {
        for(k= 0; k < c; ++k)
        {
            for(j = 0; j < w; ++j)
            {
                ((float *)dst.data)[k*w*h + i*w + j] = src_data[i*step + j*c + k]/255.;
                        
            }        
        }
    }
}

// change image order from RGB to BGR
void rgb_to_bgr(Mat im)
{
    float *data = (float *)im.data;
    for(int i = 0; i < im.cols*im.rows; ++i)
    {
        float swap = data[i];// R<=>B, save B as swap
        data[i] = data[i+im.cols*im.rows*2];// ignore G, replace B with R
        data[i+im.cols*im.rows*2] = swap; // save B in original R's place
    }
}

// used by resize_image
static float get_pixel(Mat m, int x, int y, int c)
{
    assert(x < m.cols && y < m.rows && c < m.channels());
    return ((float *)m.data)[c*m.rows*m.cols + y*m.cols + x];
}

// used by resize_image
static void set_pixel(Mat m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.cols || y >= m.rows || c >= m.channels()) return;
    assert(x < m.cols && y < m.rows && c < m.channels());
    ((float *)m.data)[c*m.rows*m.cols + y*m.cols + x] = val;
}

// used by resize_image
static void add_pixel(Mat m, int x, int y, int c, float val)
{
    assert(x < m.cols && y < m.rows && c < m.channels());
    ((float *)m.data)[c*m.rows*m.cols + y*m.cols + x] += val;
}

// resize src image to resized_img with 
//     new height (resized_img.rows) and new width (resized_img.cols)
void resize_image(Mat src, Mat resized_img)
{
    Mat part_img(src.rows, resized_img.cols, CV_32FC3);
    int r, c, k;
    float w_scale = (float)(src.cols - 1) / (resized_img.cols - 1);
    float h_scale = (float)(src.rows - 1) / (resized_img.rows - 1);
    for(k = 0; k < src.channels(); ++k)
    {
        for(r = 0; r < src.rows; ++r)
        {
            for(c = 0; c < resized_img.cols; ++c)
            {
                float val = 0; 
                if(c == resized_img.cols-1 || src.cols == 1)
                {
                    val = get_pixel(src, src.cols-1, r, k);
                } 
                else 
                {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(src, ix, r, k) + dx * get_pixel(src, ix+1, r, k);
                }
                set_pixel(part_img, c, r, k, val);
            }
        }
    }
    for(k = 0; k < src.channels(); ++k)
    {
        for(r = 0; r < resized_img.rows; ++r)
        {
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for(c = 0; c < resized_img.cols; ++c)
            {
                float val = (1-dy) * get_pixel(part_img, c, iy, k);
                set_pixel(resized_img, c, r, k, val);
            }
            if(r == resized_img.rows-1 || src.rows == 1) continue;
            for(c = 0; c < resized_img.cols; ++c)
            {
                float val = dy * get_pixel(part_img, c, iy+1, k);
                add_pixel(resized_img, c, r, k, val);
            }
        }
    }
}

// fill image with value s
void fill_image(Mat m, float s)
{
    int i;
    for(i = 0; i < m.rows*m.cols*m.channels(); i++) 
    {
        ((float *)m.data)[i] = s; 
    }
}

// embed a small image to a big image
void embed_image(Mat src, Mat dst, int dx, int dy)
{
    int x,y,k;
    for(k = 0; k < src.channels(); ++k)
    {
        printf("k = %d, dx = %d, dy = %d, src_step = %zu\n", k, dx, dy, src.step1());
        for(y = 0; y < src.rows; ++y)
        {
            for(x = 0; x < src.cols; ++x)
            {
                int src_idx = k * src.cols * src.rows + y      * src.cols + x;
                int dst_idx = k * dst.cols * dst.rows + (y+dy) * dst.cols + x + dx;
                float val =((float *)src.data)[src_idx];// HWC    get_pixel(src, x,y,k);
                ((float *)dst.data)[dst_idx] = (float)val;
            }
        }
    }
}

// print basic information of image and its top first_num pixel values
void printImage(Mat img, int first_num)
{
    printf("=== printImage ===\n");
    printf("rows:%d\n", img.rows);
    printf("cols:%d\n", img.cols);
    printf("step:%zu\n", img.step1());
    printf("channels:%d\n", img.channels());

    if (first_num)
    {
        printf("top%d:\n", first_num);
        for (int i = 0; i < first_num; i++)
        {
            printf("%d\t%f\n", i, ((float *)img.data)[i]);
        }
    }
    printf("\n");
}

int main(int argv, char ** argc)
{
    if (argv != 2) 
    {
        printf("Usage: %s INPUT_IMAGE_PATH\n", argc[0]);
        exit(-1);
    }
    Mat img = imread(argc[1]);
    if(img.data == NULL)
    {
        printf("read image error\n");
        return -1;
    }
    printf("read image successfully\n");
    printImage(img, 0);

    // convert to float, normalize with /255.
    // and change from to HWC to CHW
    printf("[convert to float]\n");
    Mat float_img(img.rows, img.cols, CV_32FC3);
    convert_to_float(img, float_img);
    printImage(float_img, 0);

    // change RGB to BGR
    printf("[RGB_to_BGR]\n");
    rgb_to_bgr(float_img);
    printImage(float_img, 0);//float_img.cols*float_img.rows*float_img.channels());

    /***********************
     *    letterbox_image  *
     **********************/
    // resize image with short side (equal scaling scale)
    printf("[resize] resize with short side\n");
    int input_w = 640; // model input
    int input_h = 480; // model input
    int img_new_w, img_new_h;
    // first caculate the new height and width
    if (((float)input_w/float_img.cols) < ((float)input_h/float_img.rows)) {
        img_new_w = input_w; 
        img_new_h = (float_img.rows * input_w)/float_img.cols;
    } 
    else 
    {
        img_new_h = input_h; 
        img_new_w = (float_img.cols * input_h)/float_img.rows;
    }
    printf("img_new_h:%d\timg_new_w:%d\n", img_new_h, img_new_w);

    Mat img_new(img_new_h, img_new_w, CV_32FC3);
    Size img_new_size(img_new_w, img_new_h);
    resize_image(float_img, img_new);

    printf("resized (equal-scaling-scaled) img_new\n");
    printImage(img_new, 0);//img_new.cols*img_new.rows*img_new.channels()); // 177*288

    // create new an image with model input shape 
    //     and fill with 0.5
    //     then embed resized image using equal-scaling-scaled to new image
    printf("[new image with model-input-shape and fill with 0.5]\n");
    Mat embeded_img(input_h, input_w, CV_32FC3);
    fill_image(embeded_img, 0.5);
    embed_image(img_new, embeded_img, (input_w-img_new_w)/2, (input_h-img_new_h)/2);
    printf("[finished preprocess]\n");
    printImage(embeded_img, 0);//embeded_img.cols*embeded_img.rows*embeded_img.channels());

    return 0;
}
