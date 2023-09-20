package com.example.open.diffusion;

import android.app.ProgressDialog;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Handler;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.AppCompatSpinner;
import androidx.fragment.app.Fragment;

import com.example.open.diffusion.core.UNet;
import com.example.open.diffusion.core.tokenizer.EngTokenizer;
import com.example.open.diffusion.core.tokenizer.TextTokenizer;

import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import ai.onnxruntime.OnnxTensor;

public class MainFragment extends Fragment {

    public static MainFragment newInstance() {

        Bundle args = new Bundle();

        MainFragment fragment = new MainFragment();
        fragment.setArguments(args);
        return fragment;
    }

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
    private View mGesView;
    private View mSetView;
    private AppCompatSpinner mSpinner;
    private View mSaveView;

    private UNet uNet;
    private Handler uiHandler;
    private TextTokenizer tokenizer;
    private boolean isWorking = false;

    @Override
    public void onDestroy() {
        super.onDestroy();
        try {
            uNet.close();
            tokenizer.close();
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        uiHandler = new Handler();
        progressDialog = new ProgressDialog(getActivity());
        progressDialog.setCancelable(false);
        uNet = new UNet(getActivity(), Device.CPU);
        tokenizer = new EngTokenizer(getActivity());

        uNet.setCallback(new UNet.Callback() {
            @Override
            public void onStep(int maxStep, int step) {
                uiHandler.post(new MyRunnable() {
                    @Override
                    public void run() {
                        mMsgView.setText(String.format("%d / %d", step + 1, maxStep));
                    }
                });
            }

            @Override
            public void onBuildImage(int status, Bitmap bitmap) {
                uiHandler.post(new MyRunnable() {
                    @Override
                    public void run() {
                        if (bitmap != null) mImageView.setImageBitmap(bitmap);
                    }
                });
            }

            @Override
            public void onComplete() {
                uiHandler.post(new MyRunnable() {
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
    }

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_main, container, false);

        mImageView = view.findViewById(R.id.image);
        mMsgView = view.findViewById(R.id.msg);
        mGuidanceView = view.findViewById(R.id.guidance);
        mStepView = view.findViewById(R.id.step);
        mPromptView = view.findViewById(R.id.prompt);
        mWidthSpinner = view.findViewById(R.id.width);
        mHeightSpinner = view.findViewById(R.id.height);
        mNetPromptView = view.findViewById(R.id.neg_prompt);
        mSeedView = view.findViewById(R.id.seed);
        mGesView = view.findViewById(R.id.generate);
        mSetView = view.findViewById(R.id.setting);
        mSpinner = view.findViewById(R.id.spinner);
        mSaveView = view.findViewById(R.id.save);

        mWidthSpinner.setSelection(3);
        mHeightSpinner.setSelection(3);

        setEnable(new File(PathManager.getModelPath(getActivity())).exists());

        view.findViewById(R.id.copy).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                progressDialog.show();
                exec.execute(new MyRunnable() {
                    @Override
                    public void run() {
                        try {
                            FileUtils.copyAssets(getActivity().getAssets(), "model", new File(PathManager.getAsssetOutputPath(getActivity())));
                        }catch (Exception e){
                            e.printStackTrace();
                        }finally {
                            uiHandler.post(new Runnable() {
                                @Override
                                public void run() {
                                    setEnable(new File(PathManager.getModelPath(getActivity())).exists());
                                    progressDialog.dismiss();
                                }
                            });
                        }
                    }
                });
            }
        });

        mSaveView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (mImageView.getDrawable() != null) {
                    try {
                        boolean success = FileUtils.saveImage(getActivity(), FileUtils.getBitmap(mImageView.getDrawable()));
                        Toast.makeText(getActivity(), success ? "保存成功 success" : "保存失败 fail", Toast.LENGTH_SHORT).show();
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }
            }
        });

        mGesView.setOnClickListener(new View.OnClickListener() {
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

        mSetView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                getActivity().getSupportFragmentManager().beginTransaction().add(R.id.container, SettingFragment.newInstance()).addToBackStack(null).commit();
            }
        });


        return view;
    }

    private MyRunnable createRunnable(){
        final String guidanceText = mGuidanceView.getText().toString();
        final String stepText = mStepView.getText().toString();
        final String prompt = mPromptView.getText().toString();
        final String negPrompt = mNetPromptView.getText().toString();
        final String seedText = mSeedView.getText().toString();

        final int num_inference_steps = TextUtils.isEmpty(stepText) ? 8 : Integer.parseInt(stepText);
        final double guidance_scale = TextUtils.isEmpty(guidanceText) ? 5f : Float.valueOf(guidanceText);
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
                    uNet.inference(seed, num_inference_steps, textEmbeddings, guidance_scale, batch_size, UNet.WIDTH, UNet.HEIGHT, mSpinner.getSelectedItemPosition());
                }catch (Exception e){
                    uiHandler.post(new Runnable() {
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

    private void setEnable(boolean isEnable){
        mGesView.setEnabled(isEnable);
        mSetView.setEnabled(isEnable);
    }
}