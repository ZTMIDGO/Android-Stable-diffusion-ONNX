package com.example.open.diffusion;

import android.content.Context;
import android.content.SharedPreferences;

public class PreferencesUtils {
    private static Context context;

    public static void init(Context c){
        context = c;
    }

    public static String getString(String key, String value){
        if (context == null){
            return null;
        }
        SharedPreferences sp=context.getSharedPreferences(context.getPackageName(), Context.MODE_PRIVATE);
        return sp.getString(key, value);
    }

    public static int getInt(String key, int value){
        if (context == null){
            return value;
        }
        SharedPreferences sp=context.getSharedPreferences(context.getPackageName(), Context.MODE_PRIVATE);
        return sp.getInt(key, value);
    }

    public static long getLong(String key, long value){
        if (context == null){
            return value;
        }
        SharedPreferences sp=context.getSharedPreferences(context.getPackageName(), Context.MODE_PRIVATE);
        return sp.getLong(key, value);
    }

    public static float getFloat(String key, float value){
        if (context == null){
            return value;
        }
        SharedPreferences sp=context.getSharedPreferences(context.getPackageName(), Context.MODE_PRIVATE);
        return sp.getFloat(key, value);
    }

    public static boolean getBoolean(String key, boolean value){
        if (context == null){
            return value;
        }
        SharedPreferences sp=context.getSharedPreferences(context.getPackageName(), Context.MODE_PRIVATE);
        return sp.getBoolean(key, value);
    }

    public static void setProperty(String key, String value){
        if (context == null){
            return;
        }

        SharedPreferences sp=context.getSharedPreferences(context.getPackageName(), Context.MODE_PRIVATE);
        SharedPreferences.Editor editor=sp.edit();
        editor.putString(key, value);
        editor.commit();
    }

    public static void setProperty(String key, float value){
        if (context == null){
            return;
        }

        SharedPreferences sp=context.getSharedPreferences(context.getPackageName(), Context.MODE_PRIVATE);
        SharedPreferences.Editor editor=sp.edit();
        editor.putFloat(key, value);
        editor.commit();
    }

    public static void setProperty(String key, long value){
        if (context == null){
            return;
        }

        SharedPreferences sp=context.getSharedPreferences(context.getPackageName(), Context.MODE_PRIVATE);
        SharedPreferences.Editor editor=sp.edit();
        editor.putLong(key, value);
        editor.commit();
    }

    public static void setProperty(String key, int value){
        if (context == null){
            return;
        }

        SharedPreferences sp=context.getSharedPreferences(context.getPackageName(), Context.MODE_PRIVATE);
        SharedPreferences.Editor editor=sp.edit();
        editor.putInt(key, value);
        editor.commit();
    }

    public static void setProperty(String key, boolean value){
        if (context == null){
            return;
        }

        SharedPreferences sp=context.getSharedPreferences(context.getPackageName(), Context.MODE_PRIVATE);
        SharedPreferences.Editor editor=sp.edit();
        editor.putBoolean(key, value);
        editor.commit();
    }

    public static void removeProperty(String key){
        if (context == null){
            return;
        }

        SharedPreferences sp=context.getSharedPreferences(context.getPackageName(), Context.MODE_PRIVATE);
        SharedPreferences.Editor editor=sp.edit();
        editor.remove(key);
        editor.commit();
    }
}
