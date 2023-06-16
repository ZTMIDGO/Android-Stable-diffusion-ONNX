package com.example.open.diffusion;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.AppCompatSpinner;

import android.app.ProgressDialog;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.open.diffusion.core.UNet;
import com.example.open.diffusion.core.tokenizer.EngTokenizer;
import com.example.open.diffusion.core.tokenizer.TextTokenizer;

import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import ai.onnxruntime.OnnxTensor;

public class MainActivity extends AppCompatActivity {
    private final ExecutorService exec = Executors.newCachedThreadPool();
    private final int[] resolution = {192, 256, 320, 384, 448, 512};

    private ImageView mImageView;
    private TextView mMsgView;
    private EditText mGuidanceView;
    private EditText mStepView;
    private EditText mPromptView;
    private EditText mNetPromptView;
    private AppCompatSpinner mWidthSpinner;
    private AppCompatSpinner mHeightSpinner;
    private ProgressDialog progressDialog;
    private EditText mSeedView;

    private UNet uNet;
    private TextTokenizer tokenizer;
    private boolean isWorking = false;

    @Override
    protected void onDestroy() {
        super.onDestroy();
        try {
            uNet.close();
            tokenizer.close();
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mImageView = findViewById(R.id.image);
        mMsgView = findViewById(R.id.msg);
        mGuidanceView = findViewById(R.id.guidance);
        mStepView = findViewById(R.id.step);
        mPromptView = findViewById(R.id.prompt);
        mWidthSpinner = findViewById(R.id.width);
        mHeightSpinner = findViewById(R.id.height);
        mNetPromptView = findViewById(R.id.neg_prompt);
        mSeedView = findViewById(R.id.seed);

        mWidthSpinner.setSelection(3);
        mHeightSpinner.setSelection(3);

        progressDialog = new ProgressDialog(MainActivity.this);

        uNet = new UNet(this, Device.CPU);
        tokenizer = new EngTokenizer(this);

        uNet.setCallback(new UNet.Callback() {
            @Override
            public void onStep(int maxStep, int step) {
                runOnUiThread(new MyRunnable() {
                    @Override
                    public void run() {
                        mMsgView.setText(String.format("%d / %d", step + 1, maxStep));
                    }
                });
            }

            @Override
            public void onBuildImage(int status, Bitmap bitmap) {
                runOnUiThread(new MyRunnable() {
                    @Override
                    public void run() {
                        if (bitmap != null) mImageView.setImageBitmap(bitmap);
                    }
                });
            }

            @Override
            public void onComplete() {
                runOnUiThread(new MyRunnable() {
                    @Override
                    public void run() {
                        mMsgView.setText("已完成");
                    }
                });
            }

            @Override
            public void onStop() {

            }
        });

        findViewById(R.id.copy).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                progressDialog.show();
                exec.execute(new MyRunnable() {
                    @Override
                    public void run() {
                        try {
                            FileUtils.copyAssets(getAssets(), "model", new File(PathManager.getAsssetOutputPath(MainActivity.this)));
                        }catch (Exception e){
                            e.printStackTrace();
                        }finally {
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    progressDialog.dismiss();
                                }
                            });
                        }
                    }
                });
            }
        });

        findViewById(R.id.generate).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    if (isWorking) return;
                    isWorking = true;
                    mMsgView.setText("初始化. . .");
                    exec.execute(createRunnable());
                }catch (Exception e){
                    e.printStackTrace();
                }
            }
        });
    }

    private MyRunnable createRunnable(){
        final String guidanceText = mGuidanceView.getText().toString();
        final String stepText = mStepView.getText().toString();
        final String prompt = mPromptView.getText().toString();
        final String negPrompt = mNetPromptView.getText().toString();
        final String seedText = mSeedView.getText().toString();
        
        final int num_inference_steps = TextUtils.isEmpty(stepText) ? 8 : Integer.parseInt(stepText);
        final double guidance_scale = TextUtils.isEmpty(guidanceText) ? 7.5f : Float.valueOf(guidanceText);
        final long seed = TextUtils.isEmpty(seedText) ? 0 : Long.parseLong(seedText);
        UNet.WIDTH = resolution[mWidthSpinner.getSelectedItemPosition()];
        UNet.HEIGHT = resolution[mHeightSpinner.getSelectedItemPosition()];

        return new MyRunnable() {
            @Override
            public void run() {
                try {
                    tokenizer.init();
                    int batch_size = 1;
                    int[] textTokenized = tokenizer.encoder(prompt);
                    int[] negTokenized = tokenizer.createUncondInput(negPrompt);

                    OnnxTensor textPromptEmbeddings = tokenizer.tensor(textTokenized);
                    OnnxTensor uncondEmbedding = tokenizer.tensor(negTokenized);
                    float[][][] textEmbeddingArray = new float[2][tokenizer.getMaxLength()][768];

                    float[] textPromptEmbeddingArray = textPromptEmbeddings.getFloatBuffer().array();
                    float[] uncondEmbeddingArray = uncondEmbedding.getFloatBuffer().array();
                    for (int i = 0; i < textPromptEmbeddingArray.length; i++)
                    {
                        textEmbeddingArray[0][i / 768][i % 768] = uncondEmbeddingArray[i];
                        textEmbeddingArray[1][i / 768][i % 768] = textPromptEmbeddingArray[i];
                    }

                    OnnxTensor textEmbeddings = OnnxTensor.createTensor(App.ENVIRONMENT, textEmbeddingArray);
                    tokenizer.close();

                    uNet.init();
                    uNet.inference(seed, num_inference_steps, textEmbeddings, guidance_scale, batch_size, UNet.WIDTH, UNet.HEIGHT);
                }catch (Exception e){
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            mMsgView.setText("Error");
                        }
                    });
                    e.printStackTrace();
                }finally {
                    isWorking = false;
                }
            }
        };
    }
}