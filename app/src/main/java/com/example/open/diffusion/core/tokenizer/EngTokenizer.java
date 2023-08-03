package com.example.open.diffusion.core.tokenizer;

import android.content.Context;
import android.text.TextUtils;
import android.util.JsonReader;
import android.util.Pair;

import com.example.open.diffusion.App;
import com.example.open.diffusion.PathManager;
import com.example.open.diffusion.StringUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.MatchResult;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtSession;

/**
 * Created by ZTMIDGO 2023/3/30
 */
public class EngTokenizer implements TextTokenizer{
    private final Pattern pattern = Pattern.compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");
    private final Map<String, Integer> encoder = new HashMap<>();
    private final Map<Integer, String> decoder = new HashMap<>();
    private final Map<Pair<String, String>, Integer> bpeRanks = new HashMap<>();

    private final String merges = "tokenizer/merges.txt";
    private final String vocab = "tokenizer/vocab.json";
    private final String model = "text_encoder/model.ort";

    private final int modelMaxLength = 77;
    private final Context context;

    private boolean isInitMap = false;
    private OrtSession session;

    public EngTokenizer(Context context) {
        this.context = context;
    }

    @Override
    public void init() throws Exception {
        if (session != null) return;
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.addConfigEntry("session.load_model_format", "ORT");
        File file = new File(PathManager.getCustomPath(context) + "/" + model);
        session = App.ENVIRONMENT.createSession(file.exists() ? file.getAbsolutePath() : PathManager.getModelPath(context) +"/" +model, options);

        if (!isInitMap){
            encoder.putAll(loadEncoder());
            decoder.putAll(loadDecoder(encoder));
            bpeRanks.putAll(loadBpeRanks());
        }
        isInitMap = true;
    }

    @Override
    public String decode(int[] ids) throws Exception {
        StringBuilder sb = new StringBuilder();
        for (int value : ids){
            if (decoder.containsKey(value)) sb.append(decoder.get(value));
        }
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < sb.length(); i++){
            String key = String.valueOf(sb.charAt(i));
            if (TokenByteUtils.BYTE_DECODER.containsKey(key)){
                result.add(TokenByteUtils.BYTE_DECODER.get(key));
            }
        }
        int[] ints = new int[result.size()];
        for (int i = 0; i < result.size(); i++) ints[i] = result.get(i);
        return new String(ints, 0, ints.length);
    }

    @Override
    public int[] encoder(String text) throws Exception {
        text = StringUtils.halfCorner(text.toLowerCase());
        List<String> stringList = new ArrayList<>();
        Matcher matcher = pattern.matcher(text);
        while (matcher.find()){
            MatchResult result = matcher.toMatchResult();
            String value = result.group().trim();
            StringBuilder sb = new StringBuilder();

            byte[] bytes = value.getBytes();
            int[] array = new int[bytes.length];
            for (int i = 0; i < array.length; i++) array[i] = bytes[i] & 0xff;

            for (int o : array){
                if (TokenByteUtils.BYTE_ENCODER.containsKey(o)){
                    sb.append(TokenByteUtils.BYTE_ENCODER.get(o));
                }
            }
            stringList.add(sb.toString());
        }

        List<String> strings = new ArrayList<>();
        for (String string : stringList){
            strings.addAll(bpe(string));
        }

        List<Integer> result = new ArrayList<>();
        result.add(49406);
        for (String word : strings){
            if (encoder.containsKey(word)) {
                result.add(encoder.get(word));
            }
        }

        int[] ids = new int[result.size()];
        for (int i = 0; i < ids.length; i++) ids[i] = result.get(i);

        int[] copy = new int[modelMaxLength];
        Arrays.fill(copy, 49407);
        System.arraycopy(ids, 0, copy, 0, ids.length < copy.length ? ids.length : copy.length);
        copy[copy.length - 1] = 49407;
        return copy;
    }

    @Override
    public OnnxTensor tensor(int[] ids) throws Exception {
        OnnxTensor input_ids = OnnxTensor.createTensor(App.ENVIRONMENT, IntBuffer.wrap(ids), new long[]{1, ids.length});
        Map<String, OnnxTensor> input = new HashMap<>();
        input.put("input_ids", input_ids);

        OrtSession.Result result = session.run(input);
        Object lastHiddenState = result.get(0).getValue();
        result.close();
        
        OnnxTensor tensor = OnnxTensor.createTensor(App.ENVIRONMENT, lastHiddenState);
        return tensor;
    }

    @Override
    public int[] createUncondInput(String text) throws Exception {
        return encoder(text);
    }

    @Override
    public int getMaxLength() {
        return modelMaxLength;
    }

    @Override
    public void close() throws Exception {
        if (session != null) session.close();

        session = null;
    }

    private List<String> bpe(String token){
        if (TextUtils.isEmpty(token)) return Arrays.asList(token);

        List<String> word = new ArrayList<>(Arrays.asList(StringUtils.toArrays(token)));
        String lastItem = word.remove(word.size() - 1);
        word.add(lastItem+"</w>");

        Set<Pair<String, String>> pairs = getPairs(word);

        while (true){
            Pair<String, String> min = null;
            int minValue = 0;
            for (Pair pair : pairs){
                if (!bpeRanks.containsKey(pair)) {
                    continue ;
                }
                int value = bpeRanks.get(pair);
                if (min == null || value < minValue){
                    min = pair;
                    minValue = value;
                }
            }

            if (min == null) break;

            int i = 0;
            List<String> newWord = new ArrayList<>();
            while (i < word.size()){
                int j = -1;
                for (int x =0; x < word.size(); x++){
                    if (x >= i && word.get(x).equals(min.first)){
                        j = x;
                        break;
                    }
                }

                if (j != -1){
                    newWord.addAll(word.subList(i, j));
                    i = j;
                }else {
                    newWord.addAll(word.subList(i, word.size()));
                    break;
                }

                if (word.get(i).equals(min.first) && i < word.size() - 1 && word.get(i + 1).equals(min.second)){
                    newWord.add(min.first + min.second);
                    i += 2;
                }else {
                    newWord.add(word.get(i));
                    i += 1;
                }
            }

            word = newWord;

            if (word.size() == 1) {
                break;
            } else {
                pairs = getPairs(word);
            }
        }

        return word;
    }

    private Set<Pair<String, String>> getPairs(List<String> word){
        Set<Pair<String, String>> result = new LinkedHashSet<>();
        for (int i =0; i < word.size() - 1; i++){
            result.add(new Pair<>(word.get(i), word.get(i + 1)));
        }
        return result;
    }

    private Map<String, Integer> loadEncoder(){
        Map<String, Integer> map = new HashMap<>();
        try {
            File file = new File(PathManager.getCustomPath(context) + "/" + vocab);
            String path = file.exists() ? file.getAbsolutePath() : PathManager.getModelPath(context) + "/" + vocab;
            JsonReader jsonReader = new JsonReader(new InputStreamReader(new FileInputStream(path)));
            jsonReader.beginObject();
            while (jsonReader.hasNext()) {
                String key = jsonReader.nextName();
                int value = jsonReader.nextInt();
                map.put(key, value);
            }
            jsonReader.close();
        }catch (Exception e){
            e.printStackTrace();
        }finally {
            return map;
        }
    }

    private Map<Integer, String> loadDecoder(Map<String, Integer> encoder){
        Map<Integer, String> result = new HashMap<>(encoder.size());
        for (Map.Entry<String, Integer> entry : encoder.entrySet()) result.put(entry.getValue(), entry.getKey());
        return result;
    }

    private Map<Pair<String, String>, Integer> loadBpeRanks(){
        Map<Pair<String, String>, Integer> result = new HashMap<>();
        try {
            File file = new File(PathManager.getCustomPath(context) + "/" + merges);
            String path = file.exists() ? file.getAbsolutePath() : PathManager.getModelPath(context) + "/" + merges;
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
            String line;
            int startLine = 1;
            int count = 0;
            while ((line = reader.readLine()) != null){
                if (startLine != 0 && startLine -- > 0) continue;
                String[] array = line.split(" ");
                if (array.length >= 2) {
                    result.put(new Pair<>(array[0], array[1]), count ++);
                }
            }
            reader.close();
        }catch (Exception e){
            e.printStackTrace();
        }finally {
            return result;
        }
    }
}
