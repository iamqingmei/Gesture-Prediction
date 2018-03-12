package ageha.globalalignment;

//import android.content.Context;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Environment;
import android.support.wearable.activity.WearableActivity;
import android.util.Log;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
//import android.widget.TextView;

public class MainActivity extends WearableActivity {


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    @Override
    protected void onStart() {
        super.onStart();

//        float[] R = new float[16], I = new float[16], earthAcc = new float[16];
//        float[] gravity_values = {-7.2522955f,0.09083997f,6.6004815f};
//
//        float[] mag_values = {8.723001f,	-6.4965f,	-51.6365f};
//        float[] deviceRelativeAcceleration = {0.4782939000000001f,-0.038179886f,-0.24776077f,0.0f};
//
//
//        SensorManager.getRotationMatrix(R, I, gravity_values, mag_values);
//
//        float[] inv = new float[16];
//        android.opengl.Matrix.invertM(inv, 0, R, 0);
//        android.opengl.Matrix.multiplyMV(earthAcc, 0, inv, 0, deviceRelativeAcceleration, 0);
//
//        Log.d("test", earthAcc[0] + ", " + earthAcc[1] + ", " + earthAcc[2] + "\n");
        InputStreamReader is = null;
        try {
            is = new InputStreamReader(getAssets().open("linear_acceleration_grav_mag.csv"));
        } catch (IOException e) {
            e.printStackTrace();
        }

        BufferedReader reader = new BufferedReader(is);
        try {
            Log.d("Acceleration", reader.readLine());
        } catch (IOException e) {
            e.printStackTrace();
        }
        String line;
        StringBuilder res = new StringBuilder();
        int c = 0;
        try {
            while ((line = reader.readLine()) != null) {
                String[] features = line.split(",");
//                tester_id,TagName,
// Time,acc_val1,acc_val2,acc_val3,acc_val4,gravity1,gravity2,gravity3,mag_val1,mag_val2,mag_val3

                float[] R = new float[16], I = new float[16], earthAcc = new float[16];
                float[] gravity_values = new float[]{Float.parseFloat(features[8]), Float.parseFloat(features[9]), Float.parseFloat(features[10])};
                float[] mag_values = new float[]{Float.parseFloat(features[11]), Float.parseFloat(features[12]), Float.parseFloat(features[13])};
                float[] deviceRelativeAcceleration = new float[]{Float.parseFloat(features[4]), Float.parseFloat(features[5]), Float.parseFloat(features[6]),Float.parseFloat(features[7])};

                SensorManager.getRotationMatrix(R, I, gravity_values, mag_values);

                float[] inv = new float[16];
//
                android.opengl.Matrix.invertM(inv, 0, R, 0);
                android.opengl.Matrix.multiplyMV(earthAcc, 0, inv, 0, deviceRelativeAcceleration, 0);

                res.append(earthAcc[0] + ", " + earthAcc[1] + ", " + earthAcc[2] + "\n");
                c++;

                if (c%50000 == 0){
                    write_to_file(res, "result" + String.valueOf(c) + ".csv");
                    res = new StringBuilder();
                    Log.d("writde","result" + String.valueOf(c) + ".csv");
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        Log.d("count", String.valueOf(c));
        Log.d("RESlength", String.valueOf(res.toString().length()));

        write_to_file(res, "result_final.csv");
    }

    /* Checks if external storage is available for read and write */
    public boolean isExternalStorageWritable() {
        String state = Environment.getExternalStorageState();
        return Environment.MEDIA_MOUNTED.equals(state);
    }

    public void write_to_file(StringBuilder content,String filename){
        if (isExternalStorageWritable()){
            // Get the directory for the user's public pictures directory.
            File file = new File(Environment.getExternalStoragePublicDirectory(
                    Environment.DIRECTORY_PICTURES), filename);
            if (!file.getParentFile().mkdirs()) {
                Log.e("dir", "Directory not created");
            }

            FileOutputStream outputStream = null;
            try {
                outputStream = new FileOutputStream(file);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }

            try {
                assert outputStream != null;
                outputStream.write(content.toString().getBytes());
                outputStream.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}
