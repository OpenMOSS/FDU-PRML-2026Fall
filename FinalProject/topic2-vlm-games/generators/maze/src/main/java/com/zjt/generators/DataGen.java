package com.zjt.generators;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.zjt.constants.MazeConstants;
import com.zjt.templates.BaseTemplate;
import com.zjt.templates.templates.*;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DataGen {
    private static java.util.Set<String> excluded;
    public static void loadExclusions(String path) {
        try { excluded = new java.util.HashSet<>(java.nio.file.Files.readAllLines(java.nio.file.Paths.get(path)));
            for (String key : excluded) if (!key.matches("[0-9a-f]{64}")) throw new IllegalArgumentException("Invalid exclusion fingerprint");
        } catch (IOException e) { throw new IllegalArgumentException("Cannot read test-state exclusion list", e); }
    }
    public static boolean isExcluded(int[][] grid) {
        if (excluded == null) throw new IllegalStateException("Load exclusions first");
        return excluded.contains(stateKey(grid));
    }
    public static String stateKey(int[][] grid) {
        StringBuilder b = new StringBuilder("maze:" + grid.length + ":" + grid[0].length + ":");
        for (int r=0;r<grid.length;r++) {
            if (r>0) b.append(';');
            for (int c=0;c<grid[r].length;c++) { if(c>0)b.append(','); b.append(grid[r][c]); }
        }
        try {
            byte[] digest=java.security.MessageDigest.getInstance("SHA-256").digest(b.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8));
            StringBuilder out=new StringBuilder(); for(byte v:digest) out.append(String.format("%02x",v & 255)); return out.toString();
        } catch (java.security.NoSuchAlgorithmException e) { throw new RuntimeException(e); }
    }


    /**
     * Generate JSON data.
     * @param maze The maze array.
     * @return A list of JSON items, where each item contains fields such as data_id, qa_type, etc.
     */
    public static List<BaseTemplate> generateJsonData(int[][] maze, int dataId) {
        List<BaseTemplate> jsonItems = new ArrayList<>();

        // Register templates
        jsonItems.add(new PlayerPosition(maze, dataId));
        jsonItems.add(new GoalPosition(maze, dataId));
        jsonItems.add(new PositionAfterMoving(maze, dataId));
        jsonItems.add(new AvailableDirections(maze, dataId));
        jsonItems.add(new FindPathToGoal(maze, dataId));
        jsonItems.add(new TurnCount(maze, dataId));

        return jsonItems;
    }

    /**
     * Generate data.
     *
     * @param idBegin   Starting ID.
     * @param dataSize  The number of data items to generate.
     * @param mazeSize  The size of the maze.
     * @param imagesDir The directory for saving images.
     * @param stateDir  The directory for saving state files.
     * @return A list of generated template data.
     */
    public static List<BaseTemplate> generateData(int idBegin, int dataSize, int mazeSize, String imagesDir, String stateDir) {
        List<BaseTemplate> data = new ArrayList<>();

        for (int id = idBegin; id < idBegin + dataSize; id++) {
            String saveImagePath = String.format("%s/image_%05d.png", imagesDir, id);
            if (excluded == null) throw new IllegalStateException("Load test exclusions before generation");
            int[][] maze;
            int retries=0;
            do {
                if (++retries>10000) throw new IllegalStateException("Test-state rejection budget exhausted");
                maze=MazeGen.generateMaze(mazeSize,mazeSize);
            } while (isExcluded(maze));
            ImageGen.drawMaze(maze, MazeConstants.CELL_SIZE, saveImagePath);
            String saveStatePath = String.format("%s/state_%05d.json", stateDir, id);
            StateGen.saveState(maze, saveStatePath);
            data.addAll(DataGen.generateJsonData(maze, id));
        }

        return data;
    }

    /**
     * Save the list of question templates as a JSON file.
     * @param questions The list of question templates.
     * @param filePath The path of the JSON file.
     */
    public static void saveDataToJson(List<BaseTemplate> questions, String filePath) {
        Gson gson = new GsonBuilder().setPrettyPrinting().create(); // Format output
        try (FileWriter writer = new FileWriter(filePath)) {
            gson.toJson(questions, writer); // Serialize the object into JSON
        } catch (IOException e) {
            throw new IllegalStateException("Failed to save dataset JSON", e);
        }
    }
}

