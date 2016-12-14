package nifi;

import static org.bytedeco.javacpp.opencv_core.CV_32SC1;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.InputStream;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import javax.imageio.ImageIO;

import org.apache.nifi.annotation.behavior.InputRequirement;
import org.apache.nifi.annotation.behavior.InputRequirement.Requirement;
import org.apache.nifi.annotation.documentation.CapabilityDescription;
import org.apache.nifi.annotation.documentation.Tags;
import org.apache.nifi.components.AllowableValue;
import org.apache.nifi.components.PropertyDescriptor;
import org.apache.nifi.flowfile.FlowFile;
import org.apache.nifi.logging.ComponentLog;
import org.apache.nifi.processor.AbstractProcessor;
import org.apache.nifi.processor.ProcessContext;
import org.apache.nifi.processor.ProcessSession;
import org.apache.nifi.processor.ProcessorInitializationContext;
import org.apache.nifi.processor.Relationship;
import org.apache.nifi.processor.exception.ProcessException;
import org.apache.nifi.processor.io.InputStreamCallback;
import org.apache.nifi.processor.util.StandardValidators;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_face;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import org.bytedeco.javacpp.presets.opencv_objdetect;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;

/**
 * A NiFi processor, which takes as input video frames coming from a video camera,
 * detects human faces in each frames and crops these detected faces.
 */
@InputRequirement(Requirement.INPUT_REQUIRED)
@Tags({"ekstream", "face", "recognition"})
@CapabilityDescription("This processor takes as input video frames with detected human faces,"
        + "and recognises these faces.")
public class FaceRecognitionProcessor extends AbstractProcessor {

    /** Allowable value. */
    public static final AllowableValue FISHER = new AllowableValue("Fisher",
            "Fisher Face Recognition", "Face recognition using the Fisher algorithm.");

    /** Allowable value. */
    public static final AllowableValue EIGEN = new AllowableValue("Eigen",
            "Eigen Face Recognition", "Face recognition using the Eigen algorithm.");

    /** Allowable value. */
    public static final AllowableValue LBPH = new AllowableValue("LBPH",
            "LBPH Face Recognition", "Face recognition using the LBPH algorithm.");

    /** Relationship "Success". */
    public static final Relationship REL_SUCCESS = new Relationship.Builder().name("success")
            .description("Video frames have been properly captured.").build();

    /** Processor property. */
    public static final PropertyDescriptor TRAINING_SET = new PropertyDescriptor.Builder()
            .name("Folder with training images.")
            .description("Specified the folder where trainaing images are located.")
            .defaultValue("test")
            .required(true)
            .addValidator(StandardValidators.NON_EMPTY_VALIDATOR)
            .build();

    /** Processor property. */
    public static final PropertyDescriptor FACE_RECOGNIZER = new PropertyDescriptor.Builder()
            .name("Face recognition algorithm.")
            .description("Specified the Face recognition algorithm to be applied.")
            .allowableValues(FISHER, EIGEN, LBPH)
            .defaultValue(FISHER.getValue())
            .required(true)
            .addValidator(StandardValidators.NON_EMPTY_VALIDATOR)
            .build();

    /** Processor property. */
    public static final PropertyDescriptor SAVE_IMAGES = new PropertyDescriptor.Builder()
            .name("Save images")
            .description("Specifies whether interim results should be saved.")
            .allowableValues(new HashSet<String>(Arrays.asList("true", "false")))
            .defaultValue("true")
            .required(true)
            .addValidator(StandardValidators.BOOLEAN_VALIDATOR)
            .build();

    /** Face recognizer. */
    private static FaceRecognizer faceRecognizer;

    /** Converter for Frames and IplImages. */
    private static OpenCVFrameConverter.ToIplImage converter;

    /** Converter for byte arrays and images. */
    private static Java2DFrameConverter flatConverter;

    /** List of processor properties. */
    private List<PropertyDescriptor> properties;

    /** List of processor relationships. */
    private Set<Relationship> relationships;

    /** Logger. */
    private ComponentLog logger;

    /**
     * {@inheritDoc}
     */
    @Override
    protected void init(final ProcessorInitializationContext context) {

        Loader.load(opencv_objdetect.class);

        logger = getLogger();

        final Set<Relationship> procRels = new HashSet<>();
        procRels.add(REL_SUCCESS);
        relationships = Collections.unmodifiableSet(procRels);

        final List<PropertyDescriptor> supDescriptors = new ArrayList<>();
        supDescriptors.add(TRAINING_SET);
        supDescriptors.add(FACE_RECOGNIZER);
        supDescriptors.add(SAVE_IMAGES);
        properties = Collections.unmodifiableList(supDescriptors);

        converter = new OpenCVFrameConverter.ToIplImage();
        flatConverter = new Java2DFrameConverter();

        logger.info("Initialision complete!");
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Set<Relationship> getRelationships() {
        return relationships;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected List<PropertyDescriptor> getSupportedPropertyDescriptors() {
        return properties;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onTrigger(final ProcessContext aContext, final ProcessSession aSession)
            throws ProcessException {

        if (null == faceRecognizer) {
            train(aContext.getProperty(TRAINING_SET).getValue(),
                    aContext.getProperty(FACE_RECOGNIZER).getValue());
        }

        FlowFile flowFile = aSession.get();
        if (flowFile == null) {
            return;
        }

        aSession.read(flowFile, new InputStreamCallback() {

            @Override
            public void process(final InputStream aStream) throws IOException {

                BufferedImage bufferedImage = ImageIO.read(aStream);
                Frame frame = toFrame(bufferedImage);
                Mat face = converter.convertToMat(frame);

                int predictedLabel = faceRecognizer.predict(face);

                logger.info("Predicted label: " + predictedLabel);

                if (aContext.getProperty(SAVE_IMAGES).asBoolean()) {
                    opencv_imgcodecs.cvSaveImage(System.currentTimeMillis() + "-reognised.png",
                            converter.convert(frame));
                }
            }
        });

        //aSession.transfer(flowFile, REL_SUCCESS);

    }

    /**
     * Trains the face recognizer.
     *
     * @param aTrainingDir directory with training images
     * @param aAlgorithm face recognition algorithm
     */
    public static void train(final String aTrainingDir, final String aAlgorithm) {

        File root = new File(aTrainingDir);

        FilenameFilter imgFilter = new FilenameFilter() {
            @Override
            public boolean accept(final File aDir, final String aName) {
                String name = aName.toLowerCase();
                return name.endsWith(".jpg") || name.endsWith(".pgm") || name.endsWith(".png");
            }
        };

        File[] imageFiles = root.listFiles(imgFilter);

        MatVector images = new MatVector(imageFiles.length);

        Mat labels = new Mat(imageFiles.length, 1, CV_32SC1);
        IntBuffer labelsBuf = labels.createBuffer();

        for (int i = 0; i < imageFiles.length; i++) {

            Mat img = opencv_imgcodecs.imread(imageFiles[i].getAbsolutePath(),
                    opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);

            int label = Integer.parseInt(imageFiles[i].getName().split("\\-")[0]);
            images.put(i, img);
            labelsBuf.put(i, label);
        }

        switch (aAlgorithm) {
        case "Fisher":
            faceRecognizer = opencv_face.createFisherFaceRecognizer();
        case "Eigen":
            faceRecognizer = opencv_face.createEigenFaceRecognizer();
        case "LBPH":
            faceRecognizer = opencv_face.createLBPHFaceRecognizer();
        default:
            faceRecognizer = opencv_face.createFisherFaceRecognizer();
        }

        faceRecognizer.train(images, labels);
    }

    /**
     * Converts an IplImage into a byte array.
     *
     * @param aImage input image
     * @return byte array
     * @throws IOException exception
     */
    public static byte[] toByteArray(final IplImage aImage) throws IOException {

        BufferedImage result = flatConverter.convert(converter.convert(aImage));
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageIO.write(result, "png", baos);
        baos.flush();
        byte[] byteImage = baos.toByteArray();
        baos.close();

        return byteImage;
    }

    /**
     * Converts a frame into a byte array.
     *
     * @param aFrame input frame
     * @return byte array
     * @throws IOException exception
     */
    public static byte[] toByteArray(final Frame aFrame) throws IOException {

        BufferedImage result = flatConverter.convert(aFrame);
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageIO.write(result, "png", baos);
        baos.flush();
        byte[] byteImage = baos.toByteArray();
        baos.close();

        return byteImage;
    }

    /**
     * Converts a buffered image into a JavaCV frame.
     *
     * @param aImage buffered image
     * @return frame
     * @throws IOException exception
     */
    public static Frame toFrame(final BufferedImage aImage) throws IOException {

        Frame result = flatConverter.convert(aImage);
        return result;
    }

    /**
     * Converts a buffered image into a JavaCV image.
     *
     * @param aImage buffered image
     * @return image
     * @throws IOException exception
     */
    public static IplImage toIplImage(final BufferedImage aImage) throws IOException {

        IplImage result = converter.convertToIplImage(flatConverter.convert(aImage));
        return result;
    }
}
