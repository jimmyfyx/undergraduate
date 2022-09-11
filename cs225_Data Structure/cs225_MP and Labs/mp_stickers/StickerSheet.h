/**
 * @file StickerSheet.h
 * Contains your declaration of the interface for the StickerSheet class.
 */
#pragma once
#include "Image.h"

class StickerSheet
{
    public:
        unsigned max_;
        Image base_image;
        unsigned *x_cor;
        unsigned *y_cor;
        Image** image;

        StickerSheet (const Image &picture, unsigned max);
        StickerSheet (const StickerSheet &other);
        ~StickerSheet ();
        const StickerSheet & operator= (const StickerSheet &other);
        void changeMaxStickers (unsigned max);
        int addSticker (Image &sticker, unsigned x, unsigned y);
        bool translate (unsigned index, unsigned x, unsigned y);
        void removeSticker (unsigned index);
        Image * getSticker (unsigned index);
        Image render () const;
};
