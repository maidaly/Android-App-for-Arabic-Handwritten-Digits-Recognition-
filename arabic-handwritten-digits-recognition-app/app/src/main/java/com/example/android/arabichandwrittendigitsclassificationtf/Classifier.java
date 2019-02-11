package com.example.android.arabichandwrittendigitsclassificationtf;

/**
 * Created by Mai Daly on 6/3/2018.
 */

//public interface for the classifer
//exposes its name and the recognize function
//which given some drawn pixels as input
//classifies what it sees as an digit image

public interface Classifier {
    String name();

    Classification recognize(final float[] pixels);
}
