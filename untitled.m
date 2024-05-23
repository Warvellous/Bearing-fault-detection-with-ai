% Fonksiyon tanımlamaları
skewns = @(x) (sum((x-mean(x)).^3)./length(x)) ./ (var(x,1).^1.5);
kurtss = @(x) (sum((x-mean(x)).^4)./length(x)) ./ (var(x,1).^2);
clearance = @(data) arrayfun(@(i) max(abs(data(:, i))) / (mean(sqrt(abs(data(:,i)))).^2), 1:size(data, 2));
shape = @(data) rms(data) ./ mean(abs(data));
impulse = @(data) max(abs(data)) ./ mean(abs(data));

% Dosya okuma
dosya_yolu = 'D:\archive\2nd_test\2nd_test'; % Okunacak dizin
dosya_listesi = dir(fullfile(dosya_yolu, '*39*')); % Dosya adında "39" geçen dosyalarin listesi

% İstatistiksel özellikleri depolamak için boş matrisler
stat_properties_rms = [];

% Verileri yükleme ve istatistiksel özellikleri hesaplama
for i = 1:numel(dosya_listesi)
    dosya_adi = dosya_listesi(i).name; % Dosya adını alma
    dosya_yolu_ve_adi = fullfile(dosya_yolu, dosya_adi); % Dosya yolunu oluşturma
    
    % Dosya okuma
    veri_struct = load(dosya_yolu_ve_adi); % Dosyayı yükleme
    veri = veri_struct.data; % Veriyi al (eğer data değişkeni içeriyorsa)

    % Zaman domainindeki veriyi frekans domainine çevirme
    veri_frekans = abs(fft(veri));
    
    % RMS değerini hesaplama
    rms_value = rms(veri_frekans);
    
    % RMS değerlerini depolama
    stat_properties_rms = [stat_properties_rms; rms_value];
end

% Eşik değeri belirleme
threshold_rms = mean(stat_properties_rms);

% Etiket matrisi oluşturma
y = stat_properties_rms > threshold_rms;

% Verileri eğitim ve test setlerine ayırma (80% eğitim, 20% test)
cv = cvpartition(size(stat_properties_rms, 1), 'HoldOut', 0.2);
idx = cv.test;

% Eğitim ve test verilerini ayırma
X_train = stat_properties_rms(~idx, :);
y_train = y(~idx, :);
X_test = stat_properties_rms(idx, :);
y_test = y(idx, :);

% SVM modelini eğitme
svm_model = fitcsvm(X_train, y_train);

% Test verisi ile tahmin yapma
y_pred = predict(svm_model, X_test);

% Doğruluk oranını hesaplama
accuracy = sum(y_pred == y_test) / length(y_test);
fprintf('Model doğruluğu: %.2f%%\n', accuracy * 100);

% Eğitim ve test sonuçlarının görselleştirilmesi
figure;
gscatter(X_train, y_train, 'rb', 'xo');
xlabel('RMS Değeri');
ylabel('Sınıf');
title('Eğitim Verisi');

figure;
gscatter(X_test, y_pred, 'rb', 'xo');
xlabel('RMS Değeri');
ylabel('Tahmin Edilen Sınıf');
title('Test Verisi');
