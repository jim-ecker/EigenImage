package edu.gatech.jecker;

/**
 * Created by jecker on 6/10/16.
 */

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;

public class Image {
    private int width;
    private int height;
    private boolean hasAlphaChannel;
    private byte[] pixels;
    private Matrix pixelMatrix;

    Image(File file) {
        BufferedImage image = null;
        try {
            image = ImageIO.read(file);
        } catch (IOException e) {
            e.printStackTrace();
        }
        assert image != null;
        pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        width = image.getWidth();
        height = image.getHeight();
        hasAlphaChannel = image.getAlphaRaster() != null;
        int pixelLength = 3;
        if (hasAlphaChannel)
            pixelLength = 4;

        this.pixelMatrix = convertTo2DWithoutUsingGetRGB(image);
    }

    private Matrix convertTo2DWithoutUsingGetRGB(BufferedImage image) {
        double[][] result = new double[this.height][this.width];
        if (this.hasAlphaChannel) {
            final int pixelLength = 4;
            for (int pixel = 0, row = 0, col = 0; pixel < this.pixels.length; pixel += pixelLength) {
                int argb = 0;
                argb += (((int) this.pixels[pixel] & 0xff) << 24); // alpha
                argb += ((int) this.pixels[pixel + 1] & 0xff); // blue
                argb += (((int) this.pixels[pixel + 2] & 0xff) << 8); // green
                argb += (((int) this.pixels[pixel + 3] & 0xff) << 16); // red
                result[row][col] = argb;
                col++;
                if (col == width) {
                    col = 0;
                    row++;
                }
            }
        } else {
            final int pixelLength = 3;
            for (int pixel = 0, row = 0, col = 0; pixel < pixels.length; pixel += pixelLength) {
                int argb = 0;
                argb += -16777216; // 255 alpha
                argb += ((int) pixels[pixel] & 0xff); // blue
                argb += (((int) pixels[pixel + 1] & 0xff) << 8); // green
                argb += (((int) pixels[pixel + 2] & 0xff) << 16); // red
                result[row][col] = (double)argb;
                col++;
                if (col == width) {
                    col = 0;
                    row++;
                }
            }
        }
        return new Matrix(result);
    }
}
