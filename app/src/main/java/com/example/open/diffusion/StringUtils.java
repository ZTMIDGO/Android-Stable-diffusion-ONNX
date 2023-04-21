package com.example.open.diffusion;

import android.text.TextUtils;

import java.math.BigDecimal;
import java.math.RoundingMode;

public class StringUtils {
    public static String link(String mark, String... args){
        StringBuilder sb = new StringBuilder();
        if (args.length == 1){
            return args[0];
        }

        for (int i = 0; i < args.length; i++){
            sb.append(args[i]);
            if (i != args.length - 1){
                sb.append(mark);
            }
        }
        return sb.toString();
    }

    public static String linkSplit(String key, String split, int size){
        StringBuilder sb = new StringBuilder();
        if (size == 1){
            return key+"=?";
        }

        for (int i = 0; i < size; i++){
            sb.append(key);
            sb.append("=?");
            if (i != size - 1){
                sb.append(split);
            }
        }
        return sb.toString();
    }

    public static String findFirstCharToUpperCase(String text){
        String item = TextUtils.isEmpty(text) ? "-" : StringUtils.toArrays(text)[0];
        return item.toUpperCase();
    }

    public static boolean isEmpty(String...strings){
        for (String string : strings){
            if (TextUtils.isEmpty(string)){
                return true;
            }
        }

        return false;
    }

    public static final String halfCorner(String str) {
        String[] regs = { "！", "，", "。", "；", "~", "《", "》", "（", "）", "？",
                "”", "｛", "｝", "“", "：", "【", "】", "”", "‘", "’", "!", ",",
                ".", ";", "`", "<", ">", "\\(", "\\)", "\\?", "'", "\\{", "}", "\"",
                ":", "\\{", "}", "\"", "\'", "\'" };
        for (int i = 0; i < regs.length / 2; i++) {
            str = str.replaceAll(regs[i], regs[i + regs.length / 2]);
        }
        return str;
    }

    public static String[] toArrays(String text){
        int[] codePoints = text.codePoints().toArray();
        String[] words = new String[codePoints.length];
        for (int i = 0; i < codePoints.length; i++){
            int code = codePoints[i];
            words[i] = new String(Character.toChars(code));
        }
        return words;
    }

    public static double round(double value, int places) {
        if (places < 0) {
            throw new IllegalArgumentException();
        }
        BigDecimal bd = new BigDecimal(value);
        bd = bd.setScale(places, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }
}
