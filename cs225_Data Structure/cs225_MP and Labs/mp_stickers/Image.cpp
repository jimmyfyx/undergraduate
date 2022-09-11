#include "Image.h"
using cs225::HSLAPixel;
using cs225::PNG;


// Lighten an Image by increasing the luminance of every pixel by 0.1.
// This function ensures that the luminance remains in the range [0, 1].
void Image::lighten()
{
    // Trace through every pixel of the image
    for (unsigned int i = 0; i < width(); i++)
    {
        for (unsigned int j = 0; j < height(); j++)
        {
            HSLAPixel & pixel = getPixel(i, j);
            // For every pixel
            // Increase the luminance by 0.1
            pixel.l = pixel.l + 0.1;
            if (pixel.l > 1)
            {
                // The luminance of every pixel cannot exceed 1
                pixel.l = 1;
            }

        }
    }
}


// Lighten an Image by increasing the luminance of every pixel by amount.
// This function ensures that the luminance remains in the range [0, 1].
void Image::lighten(double amount)
{
    // Trace through every pixel of the image
    for (unsigned int i = 0; i < width(); i++)
    {
        for (unsigned int j = 0; j < height(); j++)
        {
            HSLAPixel & pixel = getPixel(i, j);
            // For every pixel
            // Increase the luminance by amount
            pixel.l = pixel.l + amount;
            if (pixel.l > 1)
            {
                // The luminance of every pixel cannot exceed 1
                pixel.l = 1;
            }

        }
    }
}


// Darken an Image by decreasing the luminance of every pixel by 0.1.
// This function ensures that the luminance remains in the range [0, 1].
void Image::darken()
{
    // Trace through every pixel of the image
    for (unsigned int i = 0; i < width(); i++)
    {
        for (unsigned int j = 0; j < height(); j++)
        {
            HSLAPixel & pixel = getPixel(i, j);
            // For every pixel
            // Decrease the luminance by 0.1
            pixel.l = pixel.l - 0.1;
            if (pixel.l < 0)
            {
                // The luminance of every pixel cannot below 0
                pixel.l = 0;
            }

        }
    }
}


// Darken an Image by decreasing the luminance of every pixel by amount.
// This function ensures that the luminance remains in the range [0, 1].
void Image::darken(double amount)
{
    // Trace through every pixel of the image
    for (unsigned int i = 0; i < width(); i++)
    {
        for (unsigned int j = 0; j < height(); j++)
        {
            HSLAPixel & pixel = getPixel(i, j);
            // For every pixel
            // Decrease the luminance by amount
            pixel.l = pixel.l - amount;
            if (pixel.l < 0)
            {
                // The luminance of every pixel cannot below 0
                pixel.l = 0;
            }

        }
    }
}


// Desaturates an Image by decreasing the saturation of every pixel by 0.1.
// This function ensures that the saturation remains in the range [0, 1].
void Image::desaturate()
{
    // Trace through every pixel of the image
    for (unsigned int i = 0; i < width(); i++)
    {
        for (unsigned int j = 0; j < height(); j++)
        {
            HSLAPixel & pixel = getPixel(i, j);
            // For every pixel
            // Decrease the saturation by 0.1
            pixel.s = pixel.s - 0.1;
            if (pixel.l < 0)
            {
                // The saturation of every pixel cannot below 0
                pixel.l = 0;
            }

        }
    }
}


// Desaturates an Image by decreasing the saturation of every pixel by amount.
// This function ensures that the saturation remains in the range [0, 1].
void Image::desaturate(double amount)
{
    // Trace through every pixel of the image
    for (unsigned int i = 0; i < width(); i++)
    {
        for (unsigned int j = 0; j < height(); j++)
        {
            HSLAPixel & pixel = getPixel(i, j);
            // For every pixel
            // Decrease the saturation by amount
            pixel.s = pixel.s - amount;
            if (pixel.l < 0)
            {
                // The saturation of every pixel cannot below 0
                pixel.l = 0;
            }

        }
    }
}


// Saturates an Image by increasing the saturation of every pixel by 0.1.
// This function ensures that the saturation remains in the range [0, 1].
void Image::saturate()
{
    // Trace through every pixel of the image
    for (unsigned int i = 0; i < width(); i++)
    {
        for (unsigned int j = 0; j < height(); j++)
        {
            HSLAPixel & pixel = getPixel(i, j);
            // For every pixel
            // increase the saturation by 0.1
            pixel.s = pixel.s + 0.1;
            if (pixel.l > 1)
            {
                // The saturation of every pixel cannot exceed 1
                pixel.l = 1;
            }

        }
    }
}


// Saturates an Image by increasing the saturation of every pixel by amount.
// This function ensures that the saturation remains in the range [0, 1].
void Image::saturate(double amount)
{
    // Trace through every pixel of the image
    for (unsigned int i = 0; i < width(); i++)
    {
        for (unsigned int j = 0; j < height(); j++)
        {
            HSLAPixel & pixel = getPixel(i, j);
            // For every pixel
            // increase the saturation by amount
            pixel.s = pixel.s + amount;
            if (pixel.l > 1)
            {
                // The saturation of every pixel cannot exceed 1
                pixel.l = 1;
            }

        }
    }
}


// Turn the image grayscale
void Image::grayscale()
{
    // Trace through every pixel of the image
    for (unsigned int i = 0; i < width(); i++)
    {
        for (unsigned int j = 0; j < height(); j++)
        {
            HSLAPixel & pixel = getPixel(i, j);
            // For every pixel
            // Set the saturation value to 0
            pixel.s = 0;
        }
    }
}


// Rotates the color wheel by degrees.
// Rotating in a positive direction increases the degree of the hue. This function ensures that the hue remains in the range [0, 360].
void Image::rotateColor(double degrees)
{
    // Trace through every pixel of the image
    for (unsigned int i = 0; i < width(); i++)
    {
        for (unsigned int j = 0; j < height(); j++)
        {
            HSLAPixel & pixel = getPixel(i, j);
            // For every pixel
            // Increase the hue value by degrees
            pixel.h = pixel.h + degrees;
            if (pixel.h > 360)
            {   
                pixel.h = 0 + pixel.h - 360;
            }
            else if (pixel.h < 0)
            {
                pixel.h = 360 + (pixel.h - 0);
            }
        }
    }
}


// Illinify the image
void Image::illinify()
{
    // Trace through every pixel of the image
    for (unsigned x = 0; x < width(); x++) 
    {
        for (unsigned y = 0; y < height(); y++) 
        {
            HSLAPixel & pixel = getPixel(x, y);
            // For each pixel
            // Compare the distance from blue and orange color on the circle
            if (pixel.h > 293.5 || pixel.h < 113.5)
            {
                pixel.h = 11;
            }
            else
            {
                pixel.h = 216;
            }
        }
    }
}


// Scale the Image by a given factor.
void Image::scale(double factor)
{
    // First create a new PNG object (a copy of the original image)
    // Then use the member function 'resize' in PNG class to resize the origianl image
    unsigned int newWidth = (unsigned int) (factor * width());
    unsigned int newHeight = (unsigned int) (factor * height());
    PNG orig_image = PNG(*this);
    (*this).resize(newWidth, newHeight);

    // Trace through the resized image
    for (unsigned x = 0; x < newWidth; x++) 
    {
      for (unsigned y = 0; y < newHeight; y++) 
      {
        // Calculate the corresponding position(s) of the pixel(s) in the
        // original image
        unsigned orig_x = (unsigned) ((float)x / factor);
        unsigned orig_y = (unsigned) ((float)y / factor);

        // Get the original pixel(s) 
        HSLAPixel & orig_pixel = orig_image.getPixel(orig_x, orig_y);
        // Get the current pixel
        HSLAPixel & cur_pixel = (*this).getPixel(x, y);
        // Change the current pixel
        cur_pixel = orig_pixel;
      }
    }

}


// Scales the image to fit within the size (w x h).
void Image::scale(unsigned w, unsigned h)
{
    // First calculate the ratio width / height of the current image
    double ratio = (float) width() / (float) height();
    // Let height equals to h, check whether the width is out of range
    double new_width = h * ratio;
    if (new_width > w)
    {
        // The new width value is out of range
        // Let width equals to w and height equals to w / ratio
        // Calculate the scaling factor
        double factor = (float) w / (float) width();
        scale(factor);
    }
    else
    {
        // The new width value is in range
        // Width equals to h * ratio and height equals to h
        // Calculate the scaling factor
        double factor = (float) h / (float) height();
        scale(factor);
    }
}


