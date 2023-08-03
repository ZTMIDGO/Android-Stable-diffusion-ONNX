package com.example.open.diffusion;

import android.net.Uri;
import android.text.TextUtils;
import android.util.Log;

import com.google.gson.Gson;

/**
 * Created by ZTMIDGO 2023/6/20
 */
public class PreferencesManager {

    public static String getUnetName(){
        return PreferencesUtils.getString(Atts.UNET_NAME, "- - -");
    }

    public static String getTextEncoderName(){
        return PreferencesUtils.getString(Atts.TEXT_ENCODER_NAME, "- - -");
    }

    public static String getVaeDecoderName(){
        return PreferencesUtils.getString(Atts.VAE_DECODER_NAME, "- - -");
    }

    public static String getMergeName(){
        return PreferencesUtils.getString(Atts.MERGES_NAME, "- - -");
    }

    public static String getVocabName(){
        return PreferencesUtils.getString(Atts.VOCAB_NAME, "- - -");
    }
}
