# from cnn_model import BloodGroupCNN

# def main():
#     # Paths
#     train_data_path = "D:/FYProject_25-26/Blood_group_detection_using_fingerprint/dataset_prepared/train"   # change if different
#     val_data_path = "D:/FYProject_25-26/Blood_group_detection_using_fingerprint/dataset_prepared/validation"       # or None to auto-split
#     output_model = "D:/FYProject_25-26/Blood_group_detection_using_fingerprint/saved_models/bloodgroup_cnn.keras"

#     # Initialize model
#     cnn = BloodGroupCNN(input_shape=(128, 128, 1), num_classes=8)

#     # Build & compile model
#     cnn.build_model()
#     cnn.compile_model(learning_rate=0.0001)

#     # Train model
#     history = cnn.train(
#         dataset_path=train_data_path,
#         validation_data_path=val_data_path,
#         epochs=50,
#         batch_size=32,
#         output_model=output_model
#     )

#     print("\nTraining Completed Successfully!")
#     print(f"Model saved at: {output_model}")

# if __name__ == "__main__":
#     main()

from resNet_model import BloodGroupResNet50

def main():
    # Dataset paths
    train_data_path = "D:/FYProject_25-26/Blood_group_detection_using_fingerprint/dataset_prepared/train"
    val_data_path = "D:/FYProject_25-26/Blood_group_detection_using_fingerprint/dataset_prepared/validation"

    # Output model path
    output_model = "D:/FYProject_25-26/Blood_group_detection_using_fingerprint/saved_models/resnet50_bloodgroup.keras"

    # Initialize model (ResNet input shape must be 224x224x3)
    cnn = BloodGroupResNet50(input_shape=(224, 224, 3), num_classes=8)

    # Build & compile model
    cnn.build_model()
    cnn.compile_model(learning_rate=1e-4)

    # Train model
    history = cnn.train(
        dataset_path=train_data_path,
        validation_data_path=val_data_path,
        epochs=30,
        batch_size=16,
        output_model=output_model
    )

    print("\n========================")
    print("Training Completed Successfully!")
    print(f"Model saved at: {output_model}")
    print("========================")

if __name__ == "__main__":
    main()
