/* CS6300 2024 Summer A4
GT username jhan446
Some code (Coprime, Encrypt, StrToNum, NumToStr) are adapted from A3 */

package edu.gatech.seclass.sdpencryptor;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.Button;
import android.widget.EditText;
import android.widget.RadioButton;
import android.widget.TextView;
import android.view.View;
//import com.google.android.filament.View;

public class MainActivity extends AppCompatActivity {

    private EditText inputText;
    private EditText inputMultiplier;
    private EditText inputAdder;
    private TextView outputText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

    inputText = (EditText)findViewById(R.id.inputTextID);
    inputMultiplier = (EditText)findViewById(R.id.multiplierInputID);
    inputAdder = (EditText)findViewById(R.id.adderInputID);
    outputText = (TextView)findViewById(R.id.resultTextID);
    }

    public void handleClick(View view) {
        if (view.getId() == R.id.cipherButtonID) {

            String inText = inputText.getText().toString();
            if (inText.isEmpty() || !inText.matches(".*[a-zA-Z0-9].*")) {
                inputText.setError("Invalid Input Text");
            }

            int inMultiplier = Integer.parseInt((inputMultiplier.getText().toString()));
            if (inMultiplier <= 0 || inMultiplier >= 62 || !Coprime(inMultiplier, 62)) {
                inputMultiplier.setError("Invalid Multiplier Input");
            }

            int inAdder = Integer.parseInt((inputAdder.getText().toString()));
            if (inAdder < 1 || inAdder >= 62) {
                inputAdder.setError("Invalid Adder Input");
            }

            outputText.setText(Encrypt(inText, inMultiplier, inAdder));

        }
    }

    private boolean Coprime(int arg1, int y) {
        while (y != 0) {
            int helper = y;
            y = arg1 % y;
            arg1 = helper;
        }
        return arg1 == 1;
    }

    private String Encrypt(String newstr, int arg1, int arg2) {
        if (!newstr.matches(".*[a-zA-Z0-9].*") || newstr.isEmpty() || arg1 <= 0 || arg1 >= 62 || !Coprime(arg1, 62) || arg2 < 1 || arg2 >= 62) {
            return "";
        }
        StringBuilder encryptResult = new StringBuilder();
        for (char ch : newstr.toCharArray()) {
            // alphanumeric check
            if ((ch >= '0' && ch <= '9') || (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z')) {
                int helper = StrToNum(ch);
                int encrypted = (arg1 * helper + arg2) % 62;
                encryptResult.append(NumToStr(encrypted));
            } else {
                encryptResult.append(ch);
            }
        }
        return encryptResult.toString();
    }

    private int StrToNum(char ch) {
        // "0"=0, "1"=1, ..., "9"=9
        if (ch >= '0' && ch <= '9') {
            return ch - '0';
        }
        // "A"=10, "B"=12, ..., "Z"=60
        if (ch >= 'A' && ch <= 'Z') {
            return (ch - 'A') * 2 + 10;
        }
        // "a"=11, "b"=13, ..., "z"=61
        if (ch >= 'a' && ch <= 'z') {
            return (ch - 'a') * 2 + 11;
        }
        // default return
        return -1;
    }

    private char NumToStr(int num) {
        // "0"=0, "1"=1, ..., "9"=9
        if (num >= 0 && num <= 9) {
            return (char) ('0' + num);
        }
        // "A"=10, "B"=12, ..., "Z"=60
        if (num >= 10 && num < 61 && num % 2 == 0) {
            return (char) ('A' + (num - 10) / 2);
        }
        // "a"=11, "b"=13, ..., "z"=61
        if (num >= 11 && num < 62 && num % 2 != 0) {
            return (char) ('a' + (num - 11) / 2);
        }
        // default return
        return '\0';
    }
}