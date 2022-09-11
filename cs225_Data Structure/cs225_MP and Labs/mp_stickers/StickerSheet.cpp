#include "Image.h"
#include "StickerSheet.h"
#include <iostream>
using cs225::HSLAPixel;
using cs225::PNG;

// Initializes this StickerSheet with a deep copy of the base picture and the ability to 
// hold a max number of stickers (Images) with indices 0 through max - 1.
StickerSheet::StickerSheet (const Image &picture, unsigned max)
{
    // Pass in the value
    max_ = max;
    base_image = picture;
    // Defined in the PNG class, the = operator is able to make a deep copy of the Image class

    // Initialize two arrays storing x and y coordinates of added stickers
    x_cor = new unsigned[max];
    y_cor = new unsigned[max];
    for (unsigned int i = 0; i < max; i++)
    {
        x_cor[i] = 0;
        y_cor[i] = 0;
    }

    // Initialize the array holding pointers of Image with size 'max'
    // Initially, every element of the image array is NULL pointer
    image = new Image*[max];
    for (unsigned int i = 0; i < max; i++)
    {
        image[i] = NULL;
    }
}


// The copy constructor makes this StickerSheet an independent copy of the source.
StickerSheet::StickerSheet (const StickerSheet & other)
{
    // First assign values
    max_ = other.max_;
    base_image = other.base_image;

    // Initialize two arrays storing x and y coordinates of added stickers
    x_cor = new unsigned[other.max_];
    y_cor = new unsigned[other.max_];
    // Initialize the array holding pointers of Image with size 'max'
    // Initially, every element of the image array is NULL pointer
    image = new Image*[other.max_];
    // Initialize the arrays
    for (unsigned int i = 0; i < other.max_; i++)
    {
        image[i] = NULL;
        x_cor[i] = 0;
        y_cor[i] = 0;
    }

    // Count how many stickers are there in the other's image array
    unsigned int other_count = 0;
    for (unsigned int i = 0; i < other.max_; i++)
    {
        if (other.image[i] != NULL)
        {
           other_count ++;
        }
    }

    // Copy arrays storing x and y coordinates
    // Trace through the coordinates arrays
    for (unsigned int i = 0; i < other_count; i++)
    {
        // Copy every elements of 'other''s coordinates arrays to the current coordinates array
        x_cor[i] = other.x_cor[i];
        y_cor[i] = other.y_cor[i];
    }

    // Copy the elements of image array
    // Trace through other.image
    for (unsigned int i = 0; i < other_count; i++)
    {
        // Copy every Image pointer to the current image array
        image[i] = new Image();
        *(image[i]) = *(other.image[i]);
    }
}


// Destructor: Frees all space that was dynamically allocated by this StickerSheet.
StickerSheet::~StickerSheet	()
{
    // Count how many non-NULL elements are there in the image array
    unsigned int count = 0;
    for (unsigned int i = 0; i < max_; i++)
    {
        if (image[i] != NULL)
        {
           count ++;
        }
    }
    
    // Free the image array
    // First free the Image objects every element points to 
    for (unsigned int i = 0; i < max_; i++)
    {   
        delete image[i];
    }
    // Then free the entire image array
    delete image;
    
    // Free the coordinates array
    delete x_cor;
    delete y_cor;
    
}


// Adds a sticker to the StickerSheet, so that the top-left of the sticker's Image 
// is at (x, y) on the StickerSheet.
// The sticker must be added to the lowest possible layer available.
int StickerSheet::addSticker(Image & sticker, unsigned x, unsigned y)
{
    // First add the sticker pointer to the image array
    // Check whether the array is full
    // Count how many non-NULL elements are there in the image array
    unsigned int count = 0;
    for (unsigned int i = 0; i < max_; i++)
    {
        if (image[i] != NULL)
        {
           count ++;
        }
    }
    
    if (count == max_)
    {
        // The array is full
        // No more stickers can be added
        return -1;
    }

    // The array is not full
    // Find the first NULL position in the array and add
    unsigned int layer_index = 0;
    for (unsigned int i = 0; i < max_; i++)
    {
        if (image[i] == NULL)
        {
            image[i] = new Image();
            *image[i] = sticker;
            layer_index = i;
            break;
        }
    }
    
    // Store (x, y) in coordinates array
    x_cor[layer_index] = x;
    y_cor[layer_index] = y;

    return layer_index;
}


// Removes the sticker at the given zero-based layer index.
// Make sure that the other stickers don't change order.
void StickerSheet::removeSticker (unsigned index)
{
    delete image[index];
	for (unsigned i = index; i < max_ - 1; i++) 
    {
		image[i] = image[i + 1];
		x_cor[i] = x_cor[i + 1];
		y_cor[i] = y_cor[i + 1];
	}
	image[max_ - 1] = NULL;
	x_cor[max_ - 1] = 0; 
	y_cor[max_ - 1] = 0; 
}	


// Modifies the maximum number of stickers that can be 
// stored on this StickerSheet without changing existing stickers' indices.
// If the new maximum number of stickers is smaller than the current number 
// number of stickers, the stickers with indices above max - 1 will be lost.
void StickerSheet::changeMaxStickers (unsigned max)
{
    if (max == max_)
    {
        // current max equals to new max value
        // No need to change
        return;
    }

    // Count the current number of stickers added
    unsigned int count = 0;
    for (unsigned int i = 0; i < max_; i++)
    {
        if (image[i] != NULL)
        {
           count ++;
        }
    }


    // First creat a new image pointer array for more spaces with size max
    Image** new_image = new Image*[max];
    // Create new coordinates arrays
    unsigned *new_x_cor = new unsigned[max];
    unsigned *new_y_cor = new unsigned[max];
    // Initialize the arrays
    for (unsigned int i = 0; i < max; i++)
    {
        new_image[i] = NULL;
        new_x_cor[i] = 0;
        new_y_cor[i] = 0;
    }

    if (max < count)
    {
        // The max number of stickers is less than the current number of stickers
        // Discard some stickers
        // Revise the three arrays
        for (unsigned int i = 0; i < max; i++)
        {
            new_image[i] = new Image();
            *new_image[i] = *image[i];
            new_x_cor[i] = x_cor[i];
            new_y_cor[i] = y_cor[i];
        }
    }
    else if (max > count)
    {
        // No sticker pointers are lost
        // Coppy elements from old arrays to new arrays
        for (unsigned int i = 0; i < count; i++)
        {
            new_image[i] = new Image();
            *new_image[i] = *image[i];
            new_x_cor[i] = x_cor[i];
            new_y_cor[i] = y_cor[i];
        }
    }
    else if (max == count)
    {
        return;
    }

    // Free memory spaces
    // First free the objects pointed by previous image pointer array
    for (unsigned int i = 0; i < count; i++)
    {
        delete image[i];
        image[i] = NULL;
    }

    // Free memory spaces
    delete image;
    delete x_cor;
    delete y_cor;
    image = NULL;
    x_cor = NULL;
    y_cor = NULL;

    image = new_image;
    x_cor = new_x_cor;
    y_cor = new_y_cor;

    new_image = NULL;
    new_x_cor = NULL;
    new_y_cor = NULL;

    // Change the maximum number of stickers
    max_ = max; 
}	


// Returns a pointer to the sticker at the specified index, not a copy of it.
// If the index is invalid, return NULL.
Image* StickerSheet::getSticker	(unsigned index)
{
    // Check whether the index is valid
    if (index > max_ - 1)
    {
        // Index invalid
        return NULL;
    }

    return image[index];
}


// The assignment operator for the StickerSheet class.
const StickerSheet & StickerSheet::operator=(const StickerSheet & other)
{
    // Count how many stickers are there in the other's image array
    unsigned int other_count = 0;
    for (unsigned int i = 0; i < other.max_; i++)
    {
        if (other.image[i] != NULL)
        {
           other_count ++;
        }
    }

    // Count how many stickers are there in the current image array
    unsigned int count = 0;
    for (unsigned int i = 0; i < max_; i++)
    {
        if (image[i] != NULL)
        {
           count ++;
        }
    }

    // First assign values
    max_ = other.max_;
    base_image = other.base_image;

    // Create new memory spaces for the other three arrays
    Image** new_image = new Image*[other.max_];
    // Create new coordinates arrays
    unsigned *new_x_cor = new unsigned[other.max_];
    unsigned *new_y_cor = new unsigned[other.max_];
    // Initialize the arrays
    for (unsigned int i = 0; i < other.max_; i++)
    {
        new_image[i] = NULL;
        new_x_cor[i] = 0;
        new_y_cor[i] = 0;
    }

    // Copy the elements from the other's arrays
    for (unsigned int i = 0; i < other_count; i++)
    {
        new_image[i] = new Image();
        *new_image[i] = *other.image[i];
        new_x_cor[i] = other.x_cor[i];
        new_y_cor[i] = other.y_cor[i];
    }

    // Free memory spaces
    // First free the objects pointed by previous image pointer array
    for (unsigned int i = 0; i < count; i++)
    {
        delete image[i];
        image[i] = NULL;
    }

    // Free memory spaces
    delete image;
    delete x_cor;
    delete y_cor;
    image = NULL;
    x_cor = NULL;
    y_cor = NULL;

    image = new_image;
    x_cor = new_x_cor;
    y_cor = new_y_cor;

    new_image = NULL;
    new_x_cor = NULL;
    new_y_cor = NULL;
    
    return *(this);
}	



// Changes the x and y coordinates of the Image in the specified layer.
// If the layer is invalid or does not contain a sticker, this function 
// must return false. Otherwise, this function returns true.
bool StickerSheet::translate (unsigned index, unsigned x, unsigned y)
{
    // First check whether the index is valid
    if (index > max_ - 1)
    {
        // Index invalid
        return false;
    }
    // Check whether the index contains a sticker
    if (image[index] == NULL)
    {
        return false;
    }

    // Change x and y coordinates
    x_cor[index] = x;
    y_cor[index] = y;
    return true;
}


// Renders the whole StickerSheet on one Image and returns that Image.
Image StickerSheet::render () const
{
    // Set up a new Image object
    Image out_image;
    out_image = base_image;

    // Count how many stickers are added to the stickersheet
    unsigned int count_stickers = 0;
    for (unsigned int i = 0; i < max_; i++)
    {
        if (image[i] != NULL)
        {
           count_stickers ++;
        }
    }

    // First check any coordinates of the stickers are out of range of base 
    unsigned new_actual_width = out_image.width();
    unsigned new_actual_height = out_image.height();
    // For each sticker
    for (unsigned int i = 0; i < count_stickers; i++)
    {
        // For every (x, y)
        // Check either x or y (or both) is out of range
        unsigned new_width = x_cor[i] + image[i]->width();
        unsigned new_height = y_cor[i] + image[i]->height();
        
        if (new_width > new_actual_width)
        {
            new_actual_width = new_width;
        }
        if (new_height > new_actual_height)
        {
            new_actual_height = new_height;
        }
        
    }
    // Resize the output image
    out_image.resize(new_actual_width, new_actual_height);
    
    
    // Redraw the base image with stickers by layer
    // Trace through each sticker
    for (unsigned int i = 0; i < count_stickers; i++)
    {
        // For each sticker
        // Cover the base image
        for (unsigned int m = x_cor[i]; m <= x_cor[i] + image[i]->width() - 1; m++)
        {
            for (unsigned int n = y_cor[i]; n <= y_cor[i] + image[i]->height() - 1; n++)
            {
                // For every pixel
                HSLAPixel & pixel = image[i]->getPixel(m - x_cor[i], n - y_cor[i]);
                HSLAPixel & base_pixel = out_image.getPixel(m, n);
                
                // First check whether the new pixel alpha value is 0
                if (pixel.a == 0)
                {
                    // Move to the next pixel
                    continue;
                }

                // Replace the pixel of base image with new pixel
                base_pixel = pixel;
            }
        }
    }

    return out_image;
}

