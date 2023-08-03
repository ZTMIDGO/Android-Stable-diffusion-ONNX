package com.example.open.diffusion;

import android.app.ProgressDialog;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.provider.Settings;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.documentfile.provider.DocumentFile;
import androidx.fragment.app.Fragment;

import com.example.open.diffusion.core.UNet;
import com.google.gson.Gson;

import java.io.File;
import java.io.FileDescriptor;
import java.io.InputStream;
import java.util.Map;

/**
 * Created by ZTMIDGO 2023/8/3
 */
public class SettingFragment extends Fragment {
    public static SettingFragment newInstance() {

        Bundle args = new Bundle();

        SettingFragment fragment = new SettingFragment();
        fragment.setArguments(args);
        return fragment;
    }

    private TextView mUnetTextView;
    private TextView mTDTextView;
    private TextView mVaeTextView;
    private View mUnetView;
    private View mTDView;
    private View mVaeView;
    private ProgressDialog progressDialog;
    private View mMergeView;
    private TextView mMergeTextView;
    private View mVocabView;
    private TextView mVocabTextView;


    private int type = -1;
    private Handler uiHandler;
    private ActivityResultLauncher restoreLauncher;
    private ActivityResultLauncher pickLauncher;

    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        uiHandler = new Handler();
        progressDialog = new ProgressDialog(getActivity());
        progressDialog.setCancelable(false);
        initLauncher();
    }

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_setting, container, false);

        mUnetTextView = view.findViewById(R.id.unet_text);
        mTDTextView = view.findViewById(R.id.text_encoder_text);
        mVaeTextView = view.findViewById(R.id.vae_decoder_text);
        mUnetView = view.findViewById(R.id.unet);
        mTDView = view.findViewById(R.id.text_encoder);
        mVaeView = view.findViewById(R.id.vae_decoder);
        mMergeView = view.findViewById(R.id.merges);
        mMergeTextView = view.findViewById(R.id.merges_text);
        mVocabView = view.findViewById(R.id.vocab);
        mVocabTextView = view.findViewById(R.id.vocab_text);

        sync();

        mUnetView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                choice(0);
            }
        });

        mTDView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                choice(1);
            }
        });

        mVaeView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                choice(2);
            }
        });

        mMergeView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                choice(3);
            }
        });

        mVocabView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                choice(4);
            }
        });

        view.findViewById(R.id.clean).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                FileUtils.deleteAllFile(new File(PathManager.getCustomPath(getActivity())));
                sync();
            }
        });
        return view;
    }

    private void choice(int type){
        this.type = type;
        if (!PermissionHandler.checkPermissionAllGranted(getActivity(), PermissionHandler.PERMISSIONS)){
            restoreLauncher.launch(PermissionHandler.PERMISSIONS);
        }else {
            openPick();
        }
    }

    private void initLauncher() {
        restoreLauncher = PermissionHandler.createPermissionsWithArray(this, new ActivityResultCallback<Map<String, Boolean>>() {
            @Override
            public void onActivityResult(Map<String, Boolean> result) {
                for (Boolean success : result.values()) {
                    if (!success) {
                        AndroidSystem.startApplicationDetails(getActivity());
                        return;
                    }
                }
            }
        });

        pickLauncher = PermissionHandler.createPermissionsWithDocument(this, new ActivityResultCallback<Uri>() {
            @Override
            public void onActivityResult(Uri result) {
                if (result != null && type != -1){
                    try {
                        progressDialog.show();
                        DocumentFile file = DocumentFile.fromSingleUri(getActivity(), result);
                        switch (type){
                            case 0:
                                PreferencesUtils.setProperty(Atts.UNET_NAME, file.getName());
                                break;
                            case 1:
                                PreferencesUtils.setProperty(Atts.TEXT_ENCODER_NAME, file.getName());
                                break;
                            case 2:
                                PreferencesUtils.setProperty(Atts.VAE_DECODER_NAME, file.getName());
                                break;
                            case 3:
                                PreferencesUtils.setProperty(Atts.MERGES_NAME, file.getName());
                                break;
                            case 4:
                                PreferencesUtils.setProperty(Atts.VOCAB_NAME, file.getName());
                                break;
                        }
                        final InputStream in = getActivity().getContentResolver().openInputStream(result);
                        new Thread(new Runnable() {
                            @Override
                            public void run() {
                                try {
                                    String key = "";
                                    String name = "model.ort";
                                    switch (type){
                                        case 0:
                                            key = "unet";
                                            break;
                                        case 1:
                                            key = "text_encoder";
                                            break;
                                        case 2:
                                            key = "vae_decoder";
                                            break;
                                        case 3:
                                            name = "merges.txt";
                                            key = "tokenizer";
                                            break;
                                        case 4:
                                            name = "vocab.json";
                                            key = "tokenizer";
                                            break;
                                    }
                                    String path = PathManager.getCustomPath(getActivity()) + "/" + key;

                                    FileUtils.write(in, path, name);
                                }catch (Exception e){
                                    e.printStackTrace();
                                }finally {
                                    uiHandler.post(new Runnable() {
                                        @Override
                                        public void run() {
                                            sync();
                                            progressDialog.dismiss();
                                        }
                                    });
                                }
                            }
                        }).start();
                    }catch (Exception e){
                        e.printStackTrace();
                    }
                }
            }
        });
    }

    private void sync(){
        sync(0);
        sync(1);
        sync(2);
        sync(3);
        sync(4);
    }

    private void sync(int type){
        switch (type){
            case 0:
                mUnetTextView.setText(!new File(PathManager.getCustomPath(getActivity()) + "/unet/model.ort").exists() ? "- - -" : "已选择>  "+PreferencesManager.getUnetName());
                break;
            case 1:
                mTDTextView.setText(!new File(PathManager.getCustomPath(getActivity()) + "/text_encoder/model.ort").exists() ? "- - -" : "已选择>  "+PreferencesManager.getTextEncoderName());
                break;
            case 2:
                mVaeTextView.setText(!new File(PathManager.getCustomPath(getActivity()) + "/vae_decoder/model.ort").exists() ? "- - -" : "已选择>  "+PreferencesManager.getVaeDecoderName());
                break;
            case 3:
                mMergeTextView.setText(!new File(PathManager.getCustomPath(getActivity()) + "/tokenizer/merges.txt").exists() ? "- - -" : "已选择>  "+PreferencesManager.getMergeName());
                break;
            case 4:
                mVocabTextView.setText(!new File(PathManager.getCustomPath(getActivity()) + "/tokenizer/vocab.json").exists() ? "- - -" : "已选择>  "+PreferencesManager.getVocabName());
                break;
        }
    }

    private void openPick(){
        pickLauncher.launch(new String[]{"*/*"});
    }

}
